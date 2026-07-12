export const meta = {
  name: 'east-west',
  description: 'Per-construct East-vs-West F1 comparison across the CAMEL model roster — the north-star analysis',
  whenToUse: 'When you have eval CSVs for multiple models and want the East/West construct-level story: where the buckets diverge, and the shared moral-foundation collapse.',
  phases: [
    { title: 'Extract', detail: 'per-construct F1 per model in parallel' },
    { title: 'Bucket', detail: 'aggregate into East vs West and rank divergences' },
    { title: 'Critique', detail: 'adversarial check: is each claimed divergence real or a support/variance artifact?' },
  ],
}

const ROOT = '/home/szkhan/code_space3'
// args: array of { csv, model, bucket }. Default = the three v1 evals on hand.
const MODELS = (Array.isArray(args) && args.length) ? args : [
  { csv: 'eval_llama_v1.csv', model: 'Llama 3.3 70B', bucket: 'West' },
  { csv: 'eval_qwen_v1.csv', model: 'Qwen 2.5 72B', bucket: 'East' },
  { csv: 'eval_deepseek_v1.csv', model: 'DeepSeek-R1 32B', bucket: 'East' },
]
log(`East-vs-West across ${MODELS.length} models: ${MODELS.map(m => `${m.model}[${m.bucket}]`).join(', ')}`)

const PERLABEL = {
  type: 'object',
  properties: {
    model: { type: 'string' }, bucket: { type: 'string' }, macro_f1: { type: 'number' },
    labels: {
      type: 'array',
      items: {
        type: 'object',
        properties: { label: { type: 'string' }, f1: { type: 'number' }, support: { type: 'number' } },
        required: ['label', 'f1'],
      },
    },
    caveats: { type: 'string', description: 'e.g. high -1 rate makes F1 unreliable for this model' },
  },
  required: ['model', 'bucket', 'macro_f1', 'labels'],
}

phase('Extract')
const perModel = (await parallel(MODELS.map(m => () => agent(
  `Read the CAMEL eval CSV ${m.csv} at ${ROOT} for model "${m.model}" (bucket ${m.bucket}). Activate ~/myenv, use pandas. ` +
  `Report macro_f1 and per-construct f1 (+support) for all 25 constructs at prompt_id==1 (0-shot_binary) for comparability. ` +
  `If this model has a known degeneration issue (e.g. DeepSeek -1 rate), note it in caveats. Read-only.`,
  { label: `extract:${m.model}`, phase: 'Extract', schema: PERLABEL })))).filter(Boolean)

phase('Bucket')
const bucketed = await agent(
  `Aggregate CAMEL per-construct F1 into East vs West buckets. Data: ${JSON.stringify(perModel)}\n\n` +
  `For each of the 25 constructs, give mean F1 for East vs West and the divergence (East - West). Rank constructs by |divergence|. ` +
  `Separately report the six moral-foundation constructs (Care, Equality, Loyalty, Authority, Proportionality, Purity) and whether BOTH buckets collapse there ` +
  `(the emerging CLAUDE.md finding). Return a ranked divergence table and the top 5 East>West and top 5 West>East constructs. ` +
  `Down-weight any model flagged with an unreliable F1 caveat and say so.`,
  { label: 'bucket-aggregate', phase: 'Bucket' })

phase('Critique')
const critique = await agent(
  `Adversarially check this East-vs-West CAMEL analysis before it informs the thesis. Analysis: ${bucketed}\n\n` +
  `For each headline divergence claim, ask: is it driven by real signal or by (a) tiny support in one bucket, (b) a single degenerate model dragging a bucket mean, ` +
  `(c) averaging across incomparable prompt scopes? List which claims survive scrutiny and which are artifacts. Be skeptical; default to "artifact" when support is thin. ` +
  `End with the one additional data point that would most strengthen the real claims.`,
  { label: 'adversarial-critique', phase: 'Critique' })

return { perModel, bucketed, critique }
