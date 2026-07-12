# VISION.md — CAMEL: Cultural and Moral Expressions in Language

## Current Vision

### Project: CAMEL (Cultural and Moral Expressions in Language)

**Primary Stakeholder:** Shahaan Khan (MS Applied Human-Centered AI, Syracuse University iSchool). First author, driving the technical pipeline, experiments, and paper.

**Secondary Stakeholders:**
- Prof. Joshua Introne (Syracuse iSchool) — primary advisor. Weekly Thursday meetings. Owns compute coordination on OrangeGrid, confirmed vLLM stateless/prefix-caching behavior, guides research framing (ontological vs processual distinction).
- Prof. Mohammad Atari (UMass Amherst, Culture and Morality Lab) — collaborator, co-author of the CAMEL corpus paper. Directs the thesis toward what LLMs fail to capture; shepherds toward a PNAS venue. Holds pending API budget approval.

**Problem Statement:**
LLMs are increasingly used as annotators of psychological and cultural text, but it is unclear whether a model's training-culture (broadly Eastern vs Western) systematically shapes how it reads moral and cultural constructs, and whether prompt framing interacts with that culture. Existing work (Rathje et al. 2024, PNAS) shows LLMs can annotate a handful of constructs across languages but does not probe where they systematically fail on a diverse construct set, nor how cultural training origin and prompt framing interact.

**Solution:**
Benchmark Eastern and Western open-weight LLMs as annotators of 25 psychological and cultural constructs (moral foundations, cultural dimensions, behavioral markers) against a 56,796-text human-annotated gold corpus. The pipeline is config-driven, runs binary per-label YES/NO annotation via vLLM with INT4 AWQ weights and prefix caching, and evaluates with macro/micro F1 against a majority-vote binarized gold standard. Development runs on Apophis (Prof. Introne's 2x A6000 lab box); production 56K runs target the OrangeGrid HTCondor cluster (operational as of May 2026). Four research questions: RQ1 cultural variability from training corpus (main thesis), RQ2 prompt framing x cultural background (the novel "Matrix" contribution), RQ3 corpus vs post-training attribution (conditional on clean HF lineages), RQ4 architectural influence. The paper positions as a nuance extension of Rathje et al. 2024, not a refutation, ending in a practitioner decision framework for model selection by cultural target. North-star comparison is East vs West; per-model country/org origin is documented but not the analytical axis.

**Current state (as of 2026-07-06):**
OrangeGrid is stood up and validated. The V1 engine unlock (removing the fp8 KV-cache flag) holds at 224 RPM. A 100-text multi-model characterization is complete for Llama 3.3 70B and Qwen 2.5 72B, showing the central emerging finding: both collapse to near-zero F1 on the moral-foundation constructs through opposite failure modes (Llama over-predicts, Qwen under-predicts). DeepSeek-R1 32B is an active problem: it degenerates under deterministic decoding (repetition collapse at bare greedy, penalty-induced incoherence at repetition_penalty 1.15, 36.5% unparseable on the 100-text run), and its treatment is being resolved before the full corpus launches. Full 56K production has not started; it is gated on the DeepSeek decision and on diagnosing a ~27% F1 regression suspected to stem from config3's longer construct definitions.

---

## Version History

### Version 1.0 — Position Paper Against Rathje (early 2026)

**Project: CAMEL as a position/critique paper**

**Primary Stakeholder:** Shahaan Khan

**Problem Statement:**
Framed narrowly as a rebuttal to Rathje et al. 2024 (the "old news baseline") and Kristensen-McLachlan et al. 2025 (the "closest competitor"), arguing LLM annotation of psychological constructs is less reliable than claimed.

**Solution:**
A position paper leaning on early small-sample benchmark results (9-text, then 50-text) to argue the point.

**Change Reason (2026-04):**
1. Prof. Introne redirected framing away from over-indexing on Rathje as a refutation, toward a nuance extension and a practitioner decision resource.
2. The empirical work matured beyond a position piece into a full multi-model benchmark, warranting a structured research-question framing.

### Version 2.0 — Five Research Questions (mid-April 2026)

**Project: CAMEL as a multi-RQ empirical benchmark**

**Problem Statement / Solution:**
Five research questions spanning annotability hierarchy, cultural origin, prompt framing, zero-shot vs few-shot, and corpus vs post-training effects.

**Change Reason (2026-04-23):**
1. Prof. Introne consolidated five RQs into a tighter set: annotability hierarchy dropped as a standalone RQ, zero-shot vs few-shot absorbed into the prompt-framing RQ.
2. Architectural influence was retained as a fourth RQ rather than dropped, yielding the current four-RQ structure.

### Version 3.0 — Four Research Questions (current, 2026-04-23 onward)

RQ1 cultural variability from training corpus; RQ2 prompt framing x cultural background (the "Matrix," Prof's favorite); RQ3 corpus vs post-training fine-tuning (conditional); RQ4 architectural influence. This is the active framing. See WORKPLAN M7 for RQ detail and WORKLOG 2026-04-23 for the restructure rationale.
