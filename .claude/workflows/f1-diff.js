export const meta = {
  name: 'f1-diff',
  description: 'Per-construct F1 diff between two CAMEL eval CSVs to localize a regression (e.g. config2 vs config3, or 3.1 vs 3.3)',
  whenToUse: 'When macro F1 moved between two runs and you need to know WHICH of the 25 constructs drove it before burning A100 time on a full wave.',
  phases: [
    { title: 'Extract', detail: 'per-construct F1 from each eval CSV' },
    { title: 'Diff', detail: 'rank construct-level deltas and check moral-foundation collapse' },
  ],
}

// args: { a: {csv, label}, b: {csv, label} }  OR  [csvA, csvB]
let A, B
if (Array.isArray(args) && args.length >= 2) {
  A = { csv: args[0], label: args[0] }
  B = { csv: args[1], label: args[1] }
} else if (args && args.a && args.b) {
  A = { csv: args.a.csv || args.a, label: args.a.label || args.a.csv || args.a }
  B = { csv: args.b.csv || args.b, label: args.b.label || args.b.csv || args.b }
} else {
  // Sensible default: the two Eastern/Western v1 evals on hand.
  A = { csv: 'eval_llama_v1.csv', label: 'llama (West)' }
  B = { csv: 'eval_qwen_v1.csv', label: 'qwen (East)' }
  log('No args given; defaulting to eval_llama_v1.csv vs eval_qwen_v1.csv. Pass {a:{csv,label}, b:{csv,label}} to compare config2 vs config3.')
}

const ROOT = '/home/szkhan/code_space3'
log(`F1 diff: ${A.label}  vs  ${B.label}`)

const PERLABEL = {
  type: 'object',
  properties: {
    macro_f1: { type: 'number' },
    prompt_scope: { type: 'string', description: 'which prompt_id(s) this covers; note if averaged across prompts' },
    labels: {
      type: 'array',
      items: {
        type: 'object',
        properties: { label: { type: 'string' }, f1: { type: 'number' }, support: { type: 'number' } },
        required: ['label', 'f1'],
      },
    },
  },
  required: ['macro_f1', 'labels'],
}

phase('Extract')
const extract = (side) => agent(
  `Read the CAMEL eval CSV ${side.csv} at ${ROOT}. Activate ~/myenv/bin/activate, use pandas. ` +
  `Columns include label, f1, support, macro_f1, prompt_id. Return macro_f1 and a per-label f1 (with support) list covering all 25 constructs. ` +
  `If multiple prompt_ids exist, state which prompt_id you report (prefer prompt_id==1 / 0-shot_binary) so both sides are comparable. Read-only.`,
  { label: `extract:${side.label}`, phase: 'Extract', schema: PERLABEL })

const [a, b] = await parallel([() => extract(A), () => extract(B)])

phase('Diff')
const diff = await agent(
  `Compare two CAMEL per-construct F1 tables. A = ${A.label}: ${JSON.stringify(a)}\n\nB = ${B.label}: ${JSON.stringify(b)}\n\n` +
  `Produce: (1) macro F1 delta (B - A) and % relative change; (2) the constructs with the largest F1 drops and gains, ranked, as a compact table; ` +
  `(3) a specific check on the six moral-foundation constructs (Care, Equality, Loyalty, Authority, Proportionality, Purity) — do they collapse to near-zero in either run; ` +
  `(4) a one-paragraph read on whether the movement looks concentrated (a few constructs, consistent with longer config3 definitions) or diffuse (consistent with 3.1-vs-3.3 model change or sample variance). ` +
  `Flag any label where support differs between runs, since that breaks comparability.`,
  { label: 'diff', phase: 'Diff' })

return { a, b, diff }
