# ADR 003 — Evaluation Framework

## Context

Every RAG generation must be scored for quality to (a) surface regressions, (b) identify training pairs for fine-tuning, and (c) drive the quality dashboard. Two approaches:

1. **Human annotation only** — high accuracy, slow (hours/days per batch), expensive, does not scale to production traffic.
2. **Automated LLM-as-judge** — evaluates every generation in <30 seconds, enables continuous monitoring, feeds the fine-tuning pipeline automatically. Accuracy depends on the judge model and prompt quality.

## Decision

Use **automated LLM-as-judge** (RAGAS-inspired) as the primary evaluation path, with human feedback as a calibration and override mechanism.

Four metrics are computed per run:
- **Faithfulness** — claim-level grounding check. Detects hallucinations.
- **Answer Relevance** — embedding-based question generation. Detects off-topic answers.
- **Context Precision** — rank-weighted usefulness of retrieved chunks. Detects retrieval bloat.
- **Context Recall** — sentence attribution. Detects incomplete retrieval.

The judge always uses a capable non-fine-tuned model (routed via `task_type="evaluation"`) to avoid self-serving bias.

Human feedback is collected via `PATCH /runs/{run_id}/rating`. When the automated score and human rating diverge by >0.3 (on a 0–1 scale), the evaluation is flagged as `calibration_needed` and surfaced for review.

## Failure modes and detection

| Failure mode | Detection |
|---|---|
| Judge hallucinates "yes" for unsupported claims | Pearson correlation with human scores must stay >0.85. Drop triggers re-prompt. |
| Sycophantic judge prefers verbose answers | Answer relevance uses embedding similarity, not judge opinion. Resistant to length bias. |
| Judge is inconsistent | Self-consistency check: run judge 3× at temperature=0.7; flag high_variance if std >0.2. |
| Judge gaming via prompt injection in chunks | Chunks are wrapped in XML tags and the judge prompt includes an injection warning. |

## Consequences

- **Positive**: Every run scored automatically. Fine-tuning pipeline has continuous data supply. Regressions are caught within minutes.
- **Negative**: LLM-as-judge adds ~$0.01–0.05 per evaluation in API costs. Judge accuracy caps at ~90% correlation with human scores for complex analytical queries.
- **Mitigation**: Cache evaluation results; do not re-evaluate identical (query, context, answer) triples. Human review queue for `calibration_needed` evaluations.
