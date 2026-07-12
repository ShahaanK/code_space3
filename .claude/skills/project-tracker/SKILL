---
name: project-tracker
description: >
  Use this skill to create or update three project documentation files:
  VISION.md (project goal and pivot history), WORKLOG.md (session-by-session work log),
  and WORKPLAN.md (milestone and task tracker). Trigger whenever the user invokes
  /project-tracker, says "update my project docs", "log today's work", "wrap up the
  session", "initialize project tracking", "start tracking this project", or mentions
  wanting to maintain a VISION, WORKPLAN, or WORKLOG. Also trigger proactively at the
  end of a work session when the user signals they are done for the day or wants to
  record what happened. Works in two environments: in claude.ai (web/desktop/mobile),
  the docs live as files in the project knowledge and the skill emits updated versions
  as in-chat artifacts to re-upload; in a coding environment with a filesystem, the docs
  live in admin/. If all three docs already exist, run Update Mode; if any are missing,
  run Init Mode.
---

# Project Tracker

Manages three living documents:

| File | Purpose |
|---|---|
| `VISION.md` | Project goal, stakeholders, problem/solution, pivot history |
| `WORKLOG.md` | Chronological session logs, newest entry first |
| `WORKPLAN.md` | Milestone/task tracker with statuses and changelog |

---

## Step 0: Detect Environment

Where these docs live and how they are read/written depends on the environment.

- **claude.ai (web, desktop, or mobile chat)** — There is no persistent working
  filesystem and no `admin/` folder; an `admin/` directory makes no sense here. The
  three docs live as **files in the project knowledge**. Read them from the project
  files. Emit new or updated versions as **in-chat markdown artifacts** that the user
  re-uploads to the project. Past work is recovered from **past chats in this project**
  (via the conversation-search / recent-chats tools) and the current conversation, not
  from git.
- **Coding environment with a filesystem** (e.g., Claude Code) — The docs live in an
  `admin/` directory. Read and write them directly. Recover past work from `admin/`
  files plus git history.

Detect which applies: if a working filesystem with read/write access is available, use
the **filesystem path** and treat every `admin/`-prefixed reference below literally. If
running in claude.ai chat, use the **project-knowledge path** — wherever the instructions
below say `admin/VISION.md`, read instead from the project files, and instead of writing,
emit an artifact. The two-mode logic (Init vs Update) is identical in both; only the read
and write mechanics differ. Environment-specific divergences are called out inline.

---

## Step 1: Detect Mode

Check whether all three docs exist (in `admin/` on a filesystem, or in the project
knowledge in claude.ai): `VISION.md`, `WORKLOG.md`, `WORKPLAN.md`.

- **All three exist** → **Update Mode** (jump to Step 3)
- **Any are missing** → **Init Mode** (continue to Step 2)

---

## Step 2: Init Mode — Build Files From Context

The goal is to produce all three files without burdening the user. Pull from everything
available in the environment before asking a single question.

### 2a. Gather context automatically

The goal is to answer *project name*, *goal*, and *at least one milestone* before asking
the user anything. Where you look depends on the environment from Step 0.

**In claude.ai (project-knowledge path):**

This is a new run with no docs yet, so build them from the project's history. Search past
chats *in this project* and read the current conversation:

- Use the conversation-search tool with the project name and likely topic keywords to
  find prior sessions, and the recent-chats tool to sweep the last several sessions
  chronologically. Aim to reconstruct the arc: what the project is, what's been decided,
  what's been built.
- Read any project files already present (briefs, specs, design docs) — these often state
  the goal, stakeholders, and scope directly.
- Read the current conversation for the freshest direction.

Synthesize across chats rather than treating any single one as complete. If past chats
reveal a pivot (an earlier scope that was later changed), capture it for Version History.

**In a coding environment (filesystem path):**

Read whatever exists (skip silently if missing):

- `CLAUDE.md` — often has project description, team, goals
- `README.md` — project overview
- `git log --oneline -30` — reveals work done and project type
- `git log --format="%an" | sort -u` — unique contributor names
- Root directory listing, plus `src/`, `work/`, or `data/` if present — hints at domain
- Any partial files already in `admin/`

### 2b. Fall back to targeted questions only if context is sparse

If after reading all available context you still can't answer: *project name*, *goal*, and
*at least one milestone* — ask the user these four questions in a single message, not one
at a time:

1. What is this project called and what is it trying to accomplish? (one or two sentences)
2. Who is this for — who benefits most from the result?
3. What are the first 2–3 major phases or milestones? (rough is fine)
4. Solo project, or are there team members? (if team: names)

### 2c. Draft all three files

Using the gathered context, produce complete, specific drafts. Follow the **File Formats**
section exactly. Specifics matter — "Completed initial EDA on 500K NYC crime records" is
good; "Did some data work" is not. The "source of past work" below is git history on a
filesystem, or past project chats in claude.ai — use whichever applies.

- **WORKLOG**: Group the past work into meaningful sessions and create one log entry per
  cluster (one per commit-cluster on a filesystem, or one per prior chat / work session in
  claude.ai). If there is no recoverable history, create a single "Project Kickoff" entry
  dated today.
- **WORKPLAN**: Infer milestones from the gathered history and any project structure. Mark
  tasks ✅ if history shows they're done, ⏳ if recent work touches them, [ ] if only
  planned.
- **VISION**: Populate from the project brief / CLAUDE.md / README and the synthesized
  history. If a prior scope existed and was later changed (detectable from old docs, git
  messages, or an earlier chat), start a Version History.

### 2d. Show draft and wait for approval

Present all three files clearly:

```
---
Here are the three docs I'll create:

### VISION.md
[full draft]

### WORKLOG.md
[full draft]

### WORKPLAN.md
[full draft]
---

Does this look right? Tell me what to change, or approve to finalize.
```

Apply any corrections the user requests, then finalize per the environment:

- **claude.ai:** Emit all three as separate downloadable markdown artifacts (`VISION.md`,
  `WORKLOG.md`, `WORKPLAN.md`). Then tell the user plainly: *upload these three files to
  the project knowledge so future runs can find and update them.* The skill cannot write to
  project knowledge itself, so this re-upload step is what makes the docs persist.
- **Filesystem:** Write all three files to `admin/`.

---

## Step 3: Update Mode — Log This Session

### 3a. Read current state

Read all three docs (from `admin/` on a filesystem, or from the project knowledge in
claude.ai). Note:
- Which tasks are ⏳ in progress (current focus)
- The date and theme of the most recent WORKLOG entry (avoid duplicating it)
- Current project direction from VISION

If running in claude.ai and one or more docs are present in the project but you suspect
they're stale (the conversation clearly moved past what they record), still treat the
uploaded files as the baseline and layer this session's changes on top — don't silently
rebuild from scratch.

### 3b. Infer what happened this session

Look through the conversation history for signals:

- Files created, edited, or deleted (tool calls show this clearly)
- Code written, bugs fixed, features shipped
- Problems hit and whether they were resolved
- Decisions made — approach changes, abandonments, pivots
- Explicit user statements about what they accomplished or plan next
- New tasks or milestones scoped out

Also check for recent git commits if accessible (filesystem only) — they often capture
what happened most cleanly. In claude.ai, the current conversation is the primary signal;
if the session spanned earlier chats today that aren't yet logged, search past project
chats to catch them. Cross-reference with the last WORKLOG entry date to avoid re-logging
old work.

### 3c. Draft updates

**WORKLOG entry** — always add one, dated today, at the top of the file:

- **Title**: concise description of the session's main theme
- **Who**: infer from git or user statements. For a clearly solo project, you can omit
  labels or use "(Solo)". For teams, use names: "(Carmen)", "(Team)", etc.
- **Context**: one sentence on what was being worked on and why
- **Work Completed / Problems Identified / Solution Implemented**: whichever fits the session
- **Impact**: what changed, what was unblocked, or what metric moved
- **Next Steps**: only if clearly established in the conversation

**WORKPLAN changes**:

- ✅ Mark tasks complete if clearly finished this session
- ⏳ Mark tasks in-progress if started but not finished
- 🚫 Mark tasks blocked if an explicit blocker was hit
- ❌ Mark tasks abandoned if explicitly dropped
- [ ] Add new tasks for work scoped out during the session
- Add a dated changelog entry for every status change and new task

**VISION changes** — only if warranted:

- If the project's direction, stakeholder focus, or solution approach shifted — update
  Current Vision and add a new Version History entry with the change reason
- If nothing changed, leave VISION untouched (say so in the approval preview)

### 3d. Show proposed changes and wait for approval

Present the plan clearly before finalizing anything:

```
---
Here's what I'll add/update:

### WORKLOG.md — New entry at top:
[draft entry]

### WORKPLAN.md — Changes:
- M3.T2: ⏳ → ✅ (geocoding complete)
- New task: [ ] M3.T6 — [description] (Owner)
- Changelog entry dated 2025-11-05 added

### VISION.md — No changes (direction unchanged)
---

Approve to finalize, or tell me what to adjust.
```

Apply any corrections, then finalize per the environment:

- **claude.ai:** Re-emit the **full updated** doc for each file that changed, as a
  downloadable markdown artifact (not just the diff — the user needs the complete file to
  re-upload). If VISION is unchanged, say so and skip its artifact. Remind the user to
  re-upload the changed files to the project knowledge, replacing the old versions, so the
  next run reads the current state.
- **Filesystem:** Write the updated files to `admin/`.

---

## File Formats

### VISION.md

```markdown
# VISION.md

## Current Vision

### Project: [Name]

**Primary Stakeholder:** [Who benefits most — be specific about their role and need]

**Secondary Stakeholders:** 
- [Stakeholder] ([Why they care / how they use the output])
- [Stakeholder] ([Why they care / how they use the output])

**Problem Statement:**
[One or two paragraphs describing the gap or pain point. Be concrete — who experiences
this problem, when, and what they currently have to do instead.]

**Solution:**
[One or two paragraphs describing what is being built. Include key technical decisions
if they are central to the approach — e.g., "block-level predictions at 4-hour intervals"
is more useful than "a prediction model".]

---

## Version History

### Version N.0 - [Descriptive Label] (YYYY-MM-DD)

**Project: [Name at that version]**

**Primary Stakeholder:** [who]

**Problem Statement:**
[paragraph]

**Solution:**
[paragraph]

**Change Reason (YYYY-MM-DD):** [Why this version was superseded]
1. [Specific reason]
2. [Specific reason]
```

### WORKLOG.md

Newest entries at the top. Separate entries with `---`.

```markdown
# WORKLOG.md

## YYYY-MM-DD - [Session Theme] ([Who])

**Context**: [One sentence on what was being worked on and why now]

**Work Completed**:
- ([Who]) [specific thing done]
- ([Who]) [specific thing done]

**Impact**: [What changed, was unblocked, or what metric moved]

**Next Steps**: [Only include if clearly established — omit otherwise]

---
```

Use **Problems Identified** and **Solution Implemented** sections instead of (or in
addition to) Work Completed when the session was primarily about debugging or pivoting.

```markdown
**Problems Identified**:
- ([Who]) [specific problem and its consequence]

**Solution Implemented**:
- ([Who]) [what was done to resolve it]
```

Ownership labels: `(Carmen)`, `(Team)`, `(Solo)`. For solo projects, omit labels
entirely if they would be redundant throughout.

### WORKPLAN.md

```markdown
# WORKPLAN.md

## Active Plan

### Milestone N: [Name]
- [✅] MN.TN — [Description] ([Owner])
- [⏳] MN.TN — [Description] ([Owner])
- [🚫] MN.TN — [Description] ([Owner])
- [❌] MN.TN — [Description] ([Owner])
- [ ] MN.TN — [Description] ([Owner])

---

## Changelog

### YYYY-MM-DD
- ([Who]) ✅ MN.TN — [brief completion note]
- ([Who]) 🆕 MN.TN — [description of newly added task]
- ([Who]) ❌ MN.TN — [reason for abandonment]
- ([Who]) ⏳ MN.TN — [what's in flight and current status]
- ([Who]) 🚫 MN.TN — [what's blocking and why]
- ([Who]) 🔄 MN.TN — [what pivoted and why]
```

**Status key:**

| Symbol | Meaning |
|---|---|
| ✅ | Complete |
| ⏳ | In progress |
| 🚫 | Blocked |
| ❌ | Abandoned |
| `[ ]` | Not started |
| 🆕 | New task added (changelog only) |
| 🔄 | Pivoted / direction changed (changelog only) |

**Milestone IDs are stable.** Never renumber existing milestones when adding new ones —
append at the end (M5, M6, etc.) so changelog entries remain accurate.

---

## General Principles

**Be specific.** Pull actual details from the conversation and git history. "Geocoded
500K crime records using NYC GeoClient + OSM Nominatim, achieving 85% success rate" is
useful. "Made progress on geocoding" is not.

**Don't invent facts.** If you can't determine who did something, omit the owner or use
"(Team)". If you don't know the impact, describe what was completed without overstating.

**Keep entries scannable.** A WORKLOG entry should be readable in 30 seconds. Prefer
bullet points over prose. Omit sections that don't apply (e.g., skip Next Steps if none
were established).

**VISION changes are rare.** Only update VISION when the project's direction, primary
stakeholder, or core solution meaningfully shifted — not for routine progress updates.
When in doubt, leave it alone.
