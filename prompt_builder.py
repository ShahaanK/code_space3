"""
CAMEL Annotation Pipeline - Prompt Builder
============================================
Loads label definitions from config and assembles prompts by conditionally
including/excluding markers and examples sections based on prompt settings.

When a section is disabled, it is removed cleanly — no blank lines or
orphaned headers are left behind.

SYSTEM/USER SPLIT (v2 — March 2026):
  build_prompt_split() returns (system_prompt, user_prompt) for vLLM prefix
  caching. Everything that stays the same across texts for a given
  prompt × label goes into system_prompt. Only the annotated text goes
  into user_prompt. vLLM's automatic prefix caching matches on the
  identical system prefix and reuses KV states across all texts.

  build_prompt() still works as before for backward compatibility —
  it concatenates both parts into a single string.
"""

import random
import re


def select_examples(examples_list, num_examples, method="fixed", rng=None):
    """
    Select examples from a label's examples list.

    Args:
        examples_list: Full list of example strings for this label.
        num_examples: How many to select (ignored when method is "all").
        method: "fixed" (first N), "random" (randomly sampled), or "all".
        rng: random.Random instance for reproducible random selection.

    Returns:
        List of selected example strings.
    """
    if not examples_list:
        return []

    if method == "all":
        return list(examples_list)

    if num_examples <= 0:
        return []

    num_examples = min(num_examples, len(examples_list))

    if method == "random" and rng:
        return rng.sample(examples_list, num_examples)
    else:
        return examples_list[:num_examples]


def build_markers_section(label_config):
    """Build the key markers section string."""
    markers = label_config.get("markers", "").strip()
    if not markers:
        return ""
    return f"Key markers: {markers}"


def build_examples_section(label_name, examples):
    """Build the examples section string from a list of selected examples."""
    if not examples:
        return ""

    lines = [f"Example of {label_name}:"]
    for ex in examples:
        lines.append(f"  - {ex}")

    return "\n".join(lines)


def clean_prompt(text):
    """
    Clean up a prompt by collapsing multiple blank lines into at most one,
    and stripping leading/trailing whitespace.
    """
    # Collapse 3+ newlines into 2 (one blank line)
    text = re.sub(r'\n{3,}', '\n\n', text)
    return text.strip()


# =========================================================================
# SPLIT TEMPLATES
# =========================================================================
# These mirror the prompt templates in config2.yaml but are split into
# system (static per prompt×label) and user (varies per text) parts.
#
# The system template contains everything EXCEPT {text}.
# The user template contains ONLY the text and task instruction.
#
# If a prompt template is not in SPLIT_TEMPLATES, build_prompt_split()
# falls back to a generic split strategy that works for any template.
# =========================================================================


def _split_template_at_text(template):
    """
    Split a template string into (system_part, user_part) at the {text}
    placeholder. The system part is everything before === TEXT ===,
    and the user part is everything from === TEXT === onward.

    This is the generic fallback strategy that works for all current
    CAMEL prompt templates because they all have this structure:
      [role + guidelines + label block] ... === TEXT === "{text}" ... === TASK === ...

    Returns:
        (system_template, user_template) — both still contain their
        respective placeholders for .format() calls.
    """
    # Look for the TEXT section marker
    # All current templates use: === TEXT ===\n"{text}"
    text_marker = "=== TEXT ==="

    if text_marker not in template:
        # Can't split — return None to signal caller should use legacy mode
        return None

    idx = template.index(text_marker)

    system_template = template[:idx].rstrip()
    user_template = template[idx:]

    return system_template, user_template


def build_prompt_split(prompt_config, label_name, label_config, text,
                       example_selection="fixed", rng=None,
                       annotation_guidelines=""):
    """
    Build a (system_prompt, user_prompt) pair for one (prompt, label, text).

    The system_prompt is IDENTICAL for all texts with the same prompt × label.
    This enables vLLM prefix caching: the KV states for the system message
    are computed once and reused for every text.

    The user_prompt contains only the text being annotated and the task
    instruction.

    Args:
        prompt_config: Dict from config prompts list (id, name, template, etc.)
        label_name: String label name (e.g. "Loyalty").
        label_config: Dict with definition, markers, examples.
        text: The social media post to annotate.
        example_selection: "fixed", "random", or "all".
        rng: random.Random instance for reproducible selection.
        annotation_guidelines: Optional guidelines text to inject.

    Returns:
        Tuple of (system_prompt, user_prompt) strings.
        If the template can't be split, returns (None, full_prompt) where
        full_prompt is the legacy single-string format.
    """
    template = prompt_config["template"]
    definition = label_config.get("definition", "").strip()

    # --- Conditionally build sections ---
    markers_section = ""
    if prompt_config.get("include_markers", False):
        markers_section = build_markers_section(label_config)

    examples_section = ""
    if prompt_config.get("include_examples", False):
        num_examples = prompt_config.get("num_examples", 0)
        selected = select_examples(
            label_config.get("examples", []),
            num_examples,
            method=example_selection,
            rng=rng,
        )
        examples_section = build_examples_section(label_name, selected)

    # --- Try to split the template ---
    split_result = _split_template_at_text(template)

    if split_result is None:
        # Template doesn't have the expected structure — fall back to legacy
        full = template.format(
            text=text,
            label_name=label_name,
            label_definition=definition,
            markers_section=markers_section,
            examples_section=examples_section,
            annotation_guidelines=annotation_guidelines,
        )
        return None, clean_prompt(full)

    system_template, user_template = split_result

    # Fill the system part (everything except {text})
    # The system template should NOT contain {text}, but it does contain
    # all other placeholders.
    system_prompt = system_template.format(
        label_name=label_name,
        label_definition=definition,
        markers_section=markers_section,
        examples_section=examples_section,
        annotation_guidelines=annotation_guidelines,
        # Provide text as empty string in case the system template
        # somehow references it (shouldn't happen, but safe fallback)
        text="",
    )

    # Fill the user part (contains {text} and possibly {label_name})
    user_prompt = user_template.format(
        text=text,
        label_name=label_name,
        label_definition=definition,
        markers_section=markers_section,
        examples_section=examples_section,
        annotation_guidelines=annotation_guidelines,
    )

    return clean_prompt(system_prompt), clean_prompt(user_prompt)


def build_prompt(prompt_config, label_name, label_config, text,
                 example_selection="fixed", rng=None,
                 annotation_guidelines=""):
    """
    Build a complete prompt string for one (prompt, label, text) combination.

    This is the LEGACY interface — returns a single string. All existing
    scripts (benchmarks, run_annotation.py in single-string mode) continue
    to work unchanged.

    Args:
        prompt_config: Dict from config prompts list (id, name, template, etc.)
        label_name: String label name (e.g. "Loyalty").
        label_config: Dict with definition, markers, examples.
        text: The social media post to annotate.
        example_selection: "fixed", "random", or "all".
        rng: random.Random instance for reproducible selection.
        annotation_guidelines: Optional guidelines text to inject.

    Returns:
        Fully assembled prompt string ready to send to the model.
    """
    template = prompt_config["template"]
    definition = label_config.get("definition", "").strip()

    # --- Conditionally build sections ---

    # Markers section
    markers_section = ""
    if prompt_config.get("include_markers", False):
        markers_section = build_markers_section(label_config)

    # Examples section
    examples_section = ""
    if prompt_config.get("include_examples", False):
        num_examples = prompt_config.get("num_examples", 0)
        selected = select_examples(
            label_config.get("examples", []),
            num_examples,
            method=example_selection,
            rng=rng,
        )
        examples_section = build_examples_section(label_name, selected)

    # --- Fill template ---

    prompt = template.format(
        text=text,
        label_name=label_name,
        label_definition=definition,
        markers_section=markers_section,
        examples_section=examples_section,
        annotation_guidelines=annotation_guidelines,
    )

    return clean_prompt(prompt)
