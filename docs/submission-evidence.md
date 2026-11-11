# Submission evidence and benchmark prompts

This records the evidence behind the v10 submission. The Codex weight-space lab paths identify saved local artifacts. The Grok Bot paths identify reports cited in the handoff and held in its separate research checkout. [Numeric extracts for the six recovered Codex results](codex-benchmark-evidence.json)

## Azure model benchmarks

- **Task 1 paired choice.** Qwen2.5-3B scored **43/44** and Granite 4.0 1B scored **32/44** over 22 questions with both answer placements. The 22 saved `seconds` values per model give medians of **49.5165 s** for Qwen and **24.2475 s** for Granite per pair. The timer wraps evaluation of both placements. Both 4-bit checkpoints ran on Azure CPU through MLX FP32 reconstruction with six OpenBLAS threads. Saved sources are `data/task1-results/choice-summary.json`, `data/task1-results/qwen-choice.jsonl`, `data/task1-results/granite-choice.jsonl` and `eval/task1/run_baseline.py`. Granite’s **7/26** comes from a separate open-question run. [Baseline counts](../eval/baseline_counts.json)
- **Task 2 JSON constraint.** Qwen3.5-0.8B passed **30/102** and LFM3-1.7B passed **56/102**, on the same 51 field-extraction cases in both evidence orders. The Q4_K_M checkpoints ran in llama.cpp b12243, with greedy decoding, 2,048 context tokens and a 144-token response cap. Qwen’s saved schema run recorded **3.846 s p50** and **6.137 s p95** across 102 requests. A separate 16-request, two-thread LFM3 run recorded **7.024 s p50**, **10.418 s p95** and **2,053 MiB** peak server RSS for its **1.10 GB** checkpoint. These timings describe their respective request sets. LFM3 copied 190/248 fields, returned 102/102 valid arrays and missed all 14 absent fields. It returned ₹532,818 for a missing final invoice in both evidence orders. Saved sources are `data/task2-results/resumed-20261110T123700Z/schema-control/summary.json` and `data/task2-results/lfm3-20261110/` (`REPORT.md`, `schema/summary.json`, `matched-schema/summary.json`).
- **Task 2 pilot overlap.** Qwen3.5-2B passed **26/42** in free output and LFM3 passed **22/42** with the JSON constraint on those same 21 cases in two orders. The decoding settings differ, and the pilot covers only part of the 102-trial cohort. The counts were reproduced by joining `data/task2-model-upgrade-20261110T1432Z/old-paired.json` and `data/task2-results/lfm3-20261110/schema/lfm3-scored.json` by case ID and evidence order. Both run directories contain reports.

## Granite weight edits

Both edits started from separate copies of the same Granite 4.0 1B 4-bit checkpoint. The work was inspired by weight-space programming. [MIT CSAIL research](https://rikiyat.github.io/blog/weight-space-programming/)

- **MLP.** A targeted edit of `model.layers.39.mlp.down_proj.weight` changed 2,630 packed 4-bit values while preserving the affine scales and offsets. On 14 previously seen synthetic prompts with the same 32-token response cap for both checkpoints, the score fell **6/14 → 5/14**, with one loss and zero gains. The regression returned 14% instead of the historical 8%. The separate development scores were 5/7 each with unequal token caps. Saved sources are `data/task1-edit/summary.json`, `edit.json`, `dev.json` (7 prompts) and `holdout.json` (14 prompts) in the same directory, and `eval/task1/run_edit.py`.
- **Attention.** The development screen ran 416 interventions across all 16 final-layer attention heads. The selected edit attenuated head 0 of `model.layers.39.self_attn.o_proj.weight` (input columns 0–127) to the nearest representable zero. Development moved **7/13 → 8/13**. The frozen edit tied at **8/18** on fresh prompts across nine synthetic policy families, with zero gains and losses. With a 48-token cap, original/edited p50 was **10.877/10.707 s** and p95 was **24.249/32.301 s**. The timings exclude model load and were measured on two Azure CPU cores sharing VM memory bandwidth with Task 2. Saved sources are `data/task1-attention-alternative-20261110/REPORT.md`, `LATENCY_SUMMARY.json`, and `azure-artifacts/` under that directory, which contains `development.json` (13 prompts), `fresh-evaluation.json` (18 prompts), and the `one-anchor-trial/` paired outputs and reversible diff.

The attention screen found no head that fixed both target failures while preserving the controls. The selected edit corrected July from 50% to 53% on development. January stayed wrong. On the fresh set both scored 3/4 for historical dates, 0/4 for decreases, 1/2 for increases, 1/2 for revised tables, 2/2 for middle-version selection, 0/2 for missing periods and 1/2 for missing categories. The 8-bit diagnostic ended before fresh evaluation. The local Codex lab still holds the report and paired outputs copied from Azure.

The runners pass the exact `prompt` field from each fixture to Granite’s chat template. The complete case records include development and evaluation prompts, expected answers and source paths. Two examples follow. [All Granite weight-edit prompts](granite-weight-edit-prompts.json)

**MLP evaluation, `H01-1`** (expected 23%)

```text
Use the policy excerpts below. Apply the rate in force on the requested date. Reply with one percentage or UNKNOWN if the excerpts do not establish the rate.

Northbank Transit staff fare rebate. Under Circular 18 the rebate is 18%. Circular 22 replaces the 18% rebate with 23%, effective 1 April 2025.

Question: Is 18% the applicable rebate on 12 May 2025?
```

**Attention evaluation, `F01-1`** (expected 26%)

```text
Use the policy excerpts below. Apply the rate in force on the requested date. Reply with one percentage or UNKNOWN if the excerpts do not establish the rate.

Rivergate ferry duty premium. Order R1 sets 15% from 1 January 2025. Order R2 replaces 15% with 26% effective 1 October 2025.

Question: Is 15% still the premium on 2 November 2025? State the applicable percentage.
```

## Qwen weight edit

Grok Bot’s `weight-research/REPORT-v1.md` and `weight-research/phase1/LOG.md` document edits to the Qwen3.5-0.8B MLP down-projections at layers 15 and 19.

- **Direction 3, always on.** The model looped. Development dropped from **1/20 to 0/20**, and correct abstentions on the guard set fell from **13/13 to 0/13**.
- **Direction 4, first generated token only.** With the same deltas, the `P4@0.1` development score rose from **1/20 to 10/20**, and the random control stayed at **1/20**. The loops stopped. The edited model answered all **19** questions after evidence removal, so it failed the abstention guard.

The original checkpoint was retained. The sealed test stayed closed. The exact prompts, edit diff and per-case outputs were unavailable in the local Codex checkout.

## Mac and OCR measurements

- **Granite 4.0 1B on the MacBook Air M4.** Grok Bot’s `weight-research/granite60/REPORT.md` records **0.65 s** model load in session A, **9.08 s** median and **10.80 s** p90 answer latency over 53 questions in a single session, and **1.535 GB** peak MLX memory. The report points to `granite60/mac_out/` for raw logs. Later parallel rows had a **14.17 s** median under different load. The local Codex baseline independently records **0.649 s** for model load and reports separate stage timings. [Saved Mac baseline](../eval/baseline_counts.json)
- **PaddleOCR-VL 1.6, 8-bit MLX.** The Granite60 report records **549 s** to index **15 Task 2 files**, producing **150 chunks and zero recorded gaps** with the OCR model already installed. Its `granite60/score/task2_ocr_coverage.json` reports **29/33** exact matches for required answer parts. It identifies handwritten “1303” in TIL-013 as the missing gold number. These figures describe invoice answer-part coverage and file-ingest time.

## Retrieval and replay

- **Chunk sweep.** The 1,200-character target and 1,400-character hard maximum were selected on development. The top 10 contained gold evidence on **38/48** development and **18/23** held-out questions. The 18/23 figure measures evidence retrieval. [Saved sweep](../eval/chunk_sweep.json) · [Sweep code](../eval/chunk_sweep.py)
- **Product v6 retrieval.** Grok Bot’s `weight-research/task1-retrieval/DIAGNOSIS.md` reports top-10 gold-span coverage of **19/20** on development questions and **15/22** on edit-search questions. These cohorts are separate from the chunk sweep.
- **Repair replay.** The local report `eval/results/repair-candidate/gate-summary.json` records **141 evaluated**, **52/52** final frozen regression checks and **30/30** final adversarial checks. It combines `gate-final.jsonl` and `fresh-gate.jsonl`, with three targeted fresh reruns including two prompt-injection cases. The initial run copied stale indexes. The final run rebuilt the notice fixture and modified the stale file after ingest. The final counts apply to these gates within the replay. [Replay runner](../eval/product_replay.py) · [Regression IDs](../eval/floor_ids.json)
- **Other floor evaluations.** The Grok Bot handoff identifies **52/52** in `cto/floor-regression-diagnosis.md` for the v6 evaluation harness, and **42/52** for the Granite60 product path. The repair replay used its own run files and targeted reruns.

## Contract Q&A and Gemma

**Contract Q&A.** The Granite 4.0 1B Mac app passed **13/83** under strict scoring in v5 and v6, comprising **8/43** EUPL and **5/40** supplementary questions. The raw-model arm scored **11/83**. The local v5 archive is `task3-failure-dataset.zip`, with counts in `task3-failure-dataset/stats.json` and grading notes in `task3-failure-dataset/SUMMARY.md`. The v6 report cited by Grok Bot is `ldi-eval/_eval/v6/report_v6.md`.

**Gemma add-only review probe.** The Grok Bot handoff reports **299 scored held-out pairs** across four cohorts. The frozen probe uses an unchanged Gemma 3 270M instruction-tuned checkpoint at revision `ac82b4e8`. It reads 640 hidden-state values and two rule features, generates no text and preserves every rule or keyword flag.

| Cohort | Scored pairs | Meaning changes flagged, rules + keywords → with probe | Extra cosmetic checks |
|---|---:|---|---|
| Legal confirmation | 78 of 96 | 38/52 → 45/52 (p = 0.016) | 2/26 |
| Sealed legal, six families | 119 of 143 | 70/80 → 75/80 (p = 0.0625) | 2/39 |
| Sealed mixed documents | 30 | Two of two rule misses caught | See original cohort report |
| Sealed non-legal, seven families | 72 | 26/36 → 32/36 (p = 0.031) | 5/36 (14%) |

The confirmation denominator of 26 covers eligible cosmetic edits out of 44 cosmetic edits in that set. The sealed legal gain of five paired wins and zero losses fell short of the prespecified significance gate. Its AUROC was **0.871**. The high-confidence subset scored **69/71 vs 65/71** for rules and keywords (p = 0.125). The non-legal cohort exceeded the 10% extra-check target. The p values are paired at the edit level, and edits can be related within a family. The research measures agreement with RAG’s AI labels and uses reference rules. Human adjudication and validation against the app’s own rules remain pending.

| Sealed legal family | Rules + keywords | With probe |
|---|---:|---:|
| Basecamp benefits | 11/14 | 12/14 |
| GitHub general privacy | 3/5 | 4/5 |
| GitHub terms of service | 13/14 | 13/14 |
| GNU FDL | 14/15 | 15/15 |
| SEC Prosperity employment | 13/15 | 15/15 |
| SEC Wheels Up ITP | 16/17 | 16/17 |
| **Total meaning changes** | **70/80** | **75/80** |

The handoff records 99 legal training pairs, 94 legal development pairs and 124 non-legal development pairs for a retired prompt. All are outside the held-out count. It also records around **0.66 s per pair** on two Azure CPU cores in FP32, with prompts up to 286 tokens. Mac latency for this probe remains unmeasured.

The Gemma sources cited in the Grok Bot handoff are `weight-research/gemma-task2/findings.md` and `weight-research/gemma-task2/results/sealed/full_run1/sealed_report.json`. The cohort and family counts above are transcribed from the handoff. The underlying per-pair files remain in the separate research checkout. [Probe configuration and required assets](../gemma_flagger/assets/README.md)
