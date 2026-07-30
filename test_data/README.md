# test_data/

Small hand-picked sample files for cloning the pipeline to a new machine or
smoke-testing changes without touching the 56,796-text corpus on `/DATA`.
Tracked in git (unlike the full corpus, which is `.gitignore`d).

| File | Rows | Columns | Convention | Contents |
|------|------|---------|------------|----------|
| `50_random_samples.csv` | 50 | 61 | `Number`, `text` | 50-text sample + 34 raw `annotator #N` columns + 25 binarized label columns (0/1, majority-vote gold). Matches the 56K-corpus column convention. |
| `50_random_samples_ans.csv` | 50 | 62 | `Number`, `text` | Same 50 texts, raw per-annotator construct tags (comma-separated construct names per `annotator #N` cell) instead of binarized labels — the pre-binarization source for the file above. |
| `sample_prompts.xlsx` | 9 | 27 | `Text Number`, `Text` | 9-text micro-sample + 25 label columns (mostly empty — a fill-in template), used for quick prompt/config dry runs. Uses the root-`config2.yaml` column convention, NOT the `Number`/`text` convention. |
| `sample_human_annotators_ans.xlsx` | 9 | 63 | `Number`, `text` | Raw per-annotator construct tags for the same 9 texts as `sample_prompts.xlsx` (13 `annotator #N` columns populated of the format's 34-column header) — the answer key for that micro-sample. |

**Column-convention warning (see CLAUDE.md "Column Conventions"):** files here mix
`Text Number`/`Text` (root `config2.yaml`-aligned) and `Number`/`text` (56K-corpus/
`hpc/config2.yaml`/`hpc/config3.yaml`-aligned) headers. Match the file to the
config you're testing with — a mismatch silently produces wrong results rather
than an error.
