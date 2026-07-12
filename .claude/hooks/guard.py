#!/usr/bin/env python3
"""CAMEL project guardrails for Claude Code.

Enforces the "Never / Always" rules in CLAUDE.md as real hooks. Two entry modes:

  guard.py pre    PreToolUse  -- emits a permissionDecision (deny | ask | allow)
  guard.py post   PostToolUse -- validates Python/YAML edits, reports breakage

Reads the hook payload as JSON on stdin. Fails OPEN on any internal error
(exit 0, no decision) so a bug here never bricks a session -- errors are logged
to .claude/hooks/guard.log for debugging.

Enforcement tiers (see CLAUDE.md "Project Guardrails"):
  deny  -- never-do rules; no override path
  ask   -- confirm-required rules; forces an explicit permission prompt
"""
import json
import os
import re
import sys

PROJECT_ROOT = "/home/szkhan/code_space3"
LOG = os.path.join(PROJECT_ROOT, ".claude", "hooks", "guard.log")


def log(msg):
    try:
        with open(LOG, "a") as fh:
            fh.write(msg.rstrip() + "\n")
    except Exception:
        pass


def emit_pre(decision, reason):
    """Emit a PreToolUse permission decision and exit."""
    print(json.dumps({
        "hookSpecificOutput": {
            "hookEventName": "PreToolUse",
            "permissionDecision": decision,
            "permissionDecisionReason": reason,
        }
    }))
    sys.exit(0)


def emit_post_context(msg):
    """Feed a validation warning back into the model's context and exit."""
    print(json.dumps({
        "hookSpecificOutput": {
            "hookEventName": "PostToolUse",
            "additionalContext": msg,
        },
        "systemMessage": msg,
    }))
    sys.exit(0)


def norm(path):
    if not path:
        return ""
    if not os.path.isabs(path):
        path = os.path.join(PROJECT_ROOT, path)
    return os.path.normpath(path)


def has_active_fp8(content):
    """True if fp8_e4m3 appears on a line that is NOT commented out."""
    if not content or "fp8_e4m3" not in content:
        return False
    for line in content.splitlines():
        if "fp8_e4m3" in line and not line.lstrip().startswith("#"):
            return True
    return False


# ---------------------------------------------------------------------------
# PreToolUse
# ---------------------------------------------------------------------------

def check_file_edit(tool_name, ti):
    """Rules for Edit / Write / NotebookEdit."""
    path = norm(ti.get("file_path") or ti.get("notebook_path") or "")
    base = os.path.basename(path)
    # New content being written (Write.content) or inserted (Edit.new_string).
    content = ti.get("content") or ti.get("new_string") or ti.get("new_source") or ""

    # DENY: anything inside the virtualenv.
    if "/myenv/" in path + "/" or path.rstrip("/").endswith("/myenv"):
        emit_pre("deny", "CLAUDE.md: never modify files in ~/myenv (the pinned venv). Blocked.")

    # DENY: re-adding an active fp8_e4m3 flag to the HPC runner (forces slow V0 engine).
    if base == "camel_annotate_hpc.py" and has_active_fp8(content):
        emit_pre("deny",
                 "CLAUDE.md: --kv-cache-dtype fp8_e4m3 must stay commented out (keeps the "
                 "fast V1 engine). This edit re-adds it as an active flag. Blocked.")

    # ASK: master configs -- never modify without explicit confirmation.
    if base in ("config2.yaml", "config3.yaml"):
        emit_pre("ask",
                 f"CLAUDE.md: {base} is a protected master config (verbatim CAMEL definitions / "
                 "OG production config). Confirm before editing.")

    # ASK: HPC production scripts and the annotation runner (must be shown as a diff + confirmed).
    if "/hpc/" in path or base == "camel_annotate_hpc.py":
        emit_pre("ask",
                 f"CLAUDE.md: {base} is a production HPC file. Show a diff and confirm before applying.")


def split_statements(cmd):
    """Split a shell command into statements at separators, so trigger phrases
    buried inside a commit-message heredoc / echo string are not matched as if
    they were command invocations. Heuristic, not a full shell parser."""
    return re.split(r"\n|;|&&|\|\|?|\bthen\b|\bdo\b", cmd)


def check_bash(ti):
    cmd = ti.get("command") or ""

    for raw in split_statements(cmd):
        seg = raw.strip()
        if not seg:
            continue
        # strip a leading `sudo`/`time`/env-prefix so the command head is first.
        head = re.sub(r"^(sudo|time|nice|nohup|env\s+\w+=\S+)\s+", "", seg, flags=re.I)
        low = head.lower()

        # DENY: package installs / reinstalls of the pinned stack (anchored to statement head).
        if re.match(r"(pip3?|uv\s+pip|python3?\s+-m\s+pip|pip)\s+install\b", low) or \
           re.match(r"conda\s+(install|update)\b", low):
            emit_pre("deny", "CLAUDE.md: never run pip/conda install (vLLM/torch/transformers are pinned). Blocked.")

        # DENY: killing or restarting the vLLM server (kill/pkill/systemctl at statement head, targeting vllm).
        if re.match(r"(pkill|killall|kill|systemctl)\b", low) and "vllm" in low:
            emit_pre("deny", "CLAUDE.md: never stop/restart/reconfigure the vLLM server process. Blocked.")

        # git push handling (statement head is git push).
        if re.match(r"git\s+push\b", low):
            if re.search(r"(--force\b|--force-with-lease|\s-f\b|\s-\w*f\w*\b)", low):
                emit_pre("deny", "CLAUDE.md: force-push is forbidden. Blocked.")
            emit_pre("ask", "CLAUDE.md: never push to git without confirmation. Confirm this push.")

        # DENY: destructive rm/mv against data dirs or feather corpora (rm/mv at statement head).
        if re.match(r"(rm|mv)\b", low) and (
            re.search(r"/DATA(/|\b)", seg) or
            re.search(r"(^|[\s'\"/])(samples|outputs|chunks)/", seg) or
            re.search(r"\.feather\b", seg)
        ):
            emit_pre("deny",
                     "CLAUDE.md: never rm/mv files in samples/ outputs/ /DATA/ chunks/ or *.feather corpora. Blocked.")

        # DENY: writing into the venv from the shell (rm/mv/cp/redirect/tee targeting myenv/).
        if re.match(r"(rm|mv|cp|tee)\b", low) and "myenv/" in seg:
            emit_pre("deny", "CLAUDE.md: never modify ~/myenv from the shell. Blocked.")
        if re.search(r">\s*\S*myenv/", seg):
            emit_pre("deny", "CLAUDE.md: never modify ~/myenv from the shell. Blocked.")


def run_pre():
    data = json.load(sys.stdin)
    tool = data.get("tool_name", "")
    ti = data.get("tool_input", {}) or {}
    if tool in ("Edit", "Write", "NotebookEdit"):
        check_file_edit(tool, ti)
    elif tool == "Bash":
        check_bash(ti)
    # No matching rule -> allow (exit 0, no output).


# ---------------------------------------------------------------------------
# PostToolUse -- validate Python/YAML after edits (CLAUDE.md "Always").
# ---------------------------------------------------------------------------

def run_post():
    data = json.load(sys.stdin)
    ti = data.get("tool_input", {}) or {}
    resp = data.get("tool_response", {}) or {}
    path = norm(resp.get("filePath") or ti.get("file_path") or "")
    if not path or not os.path.isfile(path):
        return
    try:
        src = open(path, "r", encoding="utf-8", errors="replace").read()
    except Exception as e:
        log(f"post read fail {path}: {e}")
        return

    if path.endswith(".py"):
        import ast
        try:
            ast.parse(src)
        except SyntaxError as e:
            emit_post_context(
                f"⚠️ guard: {os.path.basename(path)} has a Python syntax error after this edit "
                f"(line {e.lineno}: {e.msg}). Fix before proceeding.")
    elif path.endswith((".yaml", ".yml")):
        try:
            import yaml
        except Exception:
            return
        try:
            yaml.safe_load(src)
        except yaml.YAMLError as e:
            emit_post_context(
                f"⚠️ guard: {os.path.basename(path)} is not valid YAML after this edit "
                f"({str(e).splitlines()[0]}). Note CLAUDE.md's apostrophe rule (group's -> group''s).")


def main():
    mode = sys.argv[1] if len(sys.argv) > 1 else "pre"
    try:
        if mode == "post":
            run_post()
        else:
            run_pre()
    except SystemExit:
        raise
    except Exception as e:
        log(f"{mode} error: {e}")
        sys.exit(0)  # fail open


if __name__ == "__main__":
    main()
