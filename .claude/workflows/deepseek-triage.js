export const meta = {
  name: 'deepseek-triage',
  description: 'Diagnose the DeepSeek-R1 -1 (unparseable) degeneration on a result CSV and recommend a next decoding config',
  whenToUse: 'When a DeepSeek run has landed and you need the -1 rate, the failure-mode breakdown, and a data-backed call on repetition_penalty vs the Option-4 exception.',
  phases: [
    { title: 'Analyze', detail: '-1 rate, failure-mode sampling, F1 impact in parallel' },
    { title: 'Synthesize', detail: 'combine into a repetition_penalty / Option-4 recommendation' },
  ],
}

// args: optional { results?: string, eval?: string }
const RESULTS = (args && args.results) || 'test_results_deepseek_v1_full.csv'
const EVALCSV = (args && args.eval) || 'eval_deepseek_v1.csv'

log(`DeepSeek triage on ${RESULTS} (eval: ${EVALCSV})`)

const STATS = {
  type: 'object',
  properties: {
    total_cells: { type: 'number' },
    neg1_count: { type: 'number' },
    neg1_rate: { type: 'number', description: 'fraction of label cells == -1' },
    per_label_neg1: {
      type: 'array',
      description: 'labels sorted by -1 rate, worst first',
      items: {
        type: 'object',
        properties: { label: { type: 'string' }, neg1_rate: { type: 'number' } },
        required: ['label', 'neg1_rate'],
      },
    },
    notes: { type: 'string' },
  },
  required: ['total_cells', 'neg1_count', 'neg1_rate', 'per_label_neg1'],
}

const MODES = {
  type: 'object',
  properties: {
    categories: {
      type: 'array',
      description: 'degeneration categories with a count and one verbatim example',
      items: {
        type: 'object',
        properties: {
          category: { type: 'string', description: 'e.g. repetition-loop, token-incoherence (whether whether), tag-salad, unterminated-think, empty, over-length-truncation' },
          approx_count: { type: 'number' },
          example: { type: 'string', description: 'short verbatim snippet of the raw response' },
        },
        required: ['category', 'approx_count', 'example'],
      },
    },
    dominant_mode: { type: 'string' },
  },
  required: ['categories', 'dominant_mode'],
}

const FIMPACT = {
  type: 'object',
  properties: {
    macro_f1: { type: 'number' },
    worst_constructs: { type: 'array', items: { type: 'string' } },
    moral_foundation_f1: { type: 'string', description: 'F1 on Care/Equality/Loyalty/Authority/Proportionality/Purity' },
  },
  required: ['macro_f1'],
}

phase('Analyze')
const [stats, modes, fimpact] = await parallel([
  () => agent(
    `Compute the -1 (unparseable) degeneration rate in the CAMEL DeepSeek result file ${RESULTS} at repo root /home/szkhan/code_space3. ` +
    `Activate the venv (source ~/myenv/bin/activate) and use pandas. The 25 label columns hold 1/0/-1 (see CLAUDE.md "Column Conventions"); ` +
    `response__<label> columns hold raw text. Count -1 cells across all 25 label columns, overall rate, and per-label -1 rate sorted worst-first. ` +
    `Do NOT modify the file. Return stats only.`,
    { label: 'neg1-stats', phase: 'Analyze', schema: STATS }),
  () => agent(
    `In the CAMEL DeepSeek result file ${RESULTS} (repo root /home/szkhan/code_space3), sample rows where a label == -1 and inspect the matching ` +
    `response__<label> raw text. Activate ~/myenv first. Categorize the degeneration modes: repetition-loop, token-level incoherence ` +
    `(e.g. "whether whether"), tag-salad, unterminated <think>, empty, and output-length truncation (max_tokens=2048). ` +
    `Give an approximate count and one short verbatim example per category, and name the dominant mode. Read-only.`,
    { label: 'failure-modes', phase: 'Analyze', schema: MODES }),
  () => agent(
    `Read the CAMEL DeepSeek per-label F1 eval ${EVALCSV} (repo root /home/szkhan/code_space3). Activate ~/myenv, use pandas. ` +
    `Report macro_f1, the worst constructs, and specifically the F1 on the six moral-foundation constructs ` +
    `(Care, Equality, Loyalty, Authority, Proportionality, Purity). Read-only.`,
    { label: 'f1-impact', phase: 'Analyze', schema: FIMPACT }),
])

phase('Synthesize')
const rec = await agent(
  `You are triaging the DeepSeek-R1-32B deterministic-annotation blocker (CLAUDE.md "Active Problems"). Context: bare greedy loops; ` +
  `greedy + repetition_penalty 1.15 gave ~36.5% -1 on a prior 100-text run. Current MODEL_OVERRIDES: max_tokens=2048, repetition_penalty=1.15, temperature=0.\n\n` +
  `-1 stats: ${JSON.stringify(stats)}\n\nFailure modes: ${JSON.stringify(modes)}\n\nF1 impact: ${JSON.stringify(fimpact)}\n\n` +
  `Decide: (a) does a gentler repetition_penalty (~1.05) plausibly help given the dominant failure mode, or (b) is this the point to escalate to the ` +
  `June diagnosis "Option 4" (document DeepSeek as a reasoning-model exception, or swap the Eastern reasoning slot for a controllable non-thinking model)? ` +
  `Do NOT recommend launching the full corpus for DeepSeek. Give a crisp recommendation with the single most informative next experiment and the exact ` +
  `MODEL_OVERRIDES change it implies (as a proposal only — editing camel_annotate_hpc.py requires Shahaan's confirmation).`,
  { label: 'recommendation', phase: 'Synthesize' })

return { stats, modes, fimpact, recommendation: rec }
