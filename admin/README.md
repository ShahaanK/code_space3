# admin/

Project management and operational docs for CAMEL. Index below; see each file
for detail.

| File | What it is | Maintained by |
|------|-----------|---------------|
| `VISION.md` | One-page project vision: stakeholders, problem statement, solution summary, current state. Point new readers (or a new Claude session) here first. | Manual edits |
| `WORKLOG.md` | Reverse-chronological narrative of every work session — what was tried, what broke, what was decided and why. Latest session on top. | `/project-tracker` — do not hand-edit |
| `WORKPLAN.md` | Milestone/task tracking (checkboxes), reference tables, changelog. | `/project-tracker` — do not hand-edit |
| `OrangeGrid_Quickstart.md` | OrangeGrid (HTCondor) connection, environment, job submission, and cluster-quirk reference — login node, ClassAd gotchas, GPU pool, HF cache bind pattern. | Manual edits |
| `Apophis_Jupyter_Terminal_Recovery.md` | Operational note on the 2026-07-23 Jupyter Lab terminal-hang incident on Apophis (port 8891) and the recovery steps, kept in case the problem recurs on the long-lived server. | Manual edits |
| `old/` | Superseded WORKLOG/WORKPLAN snapshots, archived when the live files are regenerated or split by date range. See `old/README.MD` (one line: "This is where all the old files get moved to"). Each archived file's own filename/date range indicates why the live file supersedes it — don't delete, they're the audit trail. | Append-only archive |

CLAUDE.md (repo root, local to Apophis) directs that at the start of every
session `VISION.md`, `WORKLOG.md`, and `WORKPLAN.md` be read in full before any
other work, and that `/project-tracker` be run at the end of every session to
update `WORKLOG.md`/`WORKPLAN.md`. Do not hand-edit those two files outside
that workflow.
