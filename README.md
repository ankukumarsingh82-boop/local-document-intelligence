# Local document intelligence

Ask questions about text and PDF files on this Mac. Answers are computed here.

`ldi setup` is the one command that uses the network, and only to download weights. `ldi ask` loads those weights with the hub offline.

## Before the first answer

1. Restart the Mac if it has been awake for a long time.
2. Plug in.
3. Quit other apps. Keep one Cursor window, with indexing paused.
4. Run `ldi status`. Load a model when it says the 3B model fits.

The Air will get warm on a long answer. That is the M4 slowing itself.

## Setup

```bash
uv sync
uv run ldi status
uv run ldi ingest documents
uv run ldi search "replacement seal"
uv run ldi setup
uv run ldi ask "What is the replacement seal part?"
```

`uv sync` installs MLX. It does not download a model.

## Models

| Command | Weights | Metal cap |
|---|---|---|
| `ldi setup` / `ldi ask` | Qwen2.5 3B Instruct, 4-bit | 4 GB |
| `ldi setup --model 7b` / `ldi ask --model 7b` | Qwen2.5 7B Instruct, 4-bit | 7 GB |
| `ldi setup --model 1b` | Granite 4.0 1B, 4-bit | 1.2 GB, from a measured peak |
| `ldi setup --model bge` | BGE Small English v1.5, 8-bit | 512 MB. `ldi embed` does not load it |
| `ldi setup --model ocr` | PaddleOCR-VL 1.6, 8-bit | 2 GB. A scanned page loads it only while the chat model is unloaded |

Start with 3B. Move to 7B when `ldi status` says it fits. The tool will not download a larger model.

## Commands

| Command | What it does |
|---|---|
| `ldi status` | Disk, memory pressure, and whether a load is allowed |
| `ldi ingest <path>` | Index text and PDF files. A scanned page uses the OCR model, then unloads it |
| `ldi search "<question>"` | Show matching passages |
| `ldi setup` | Download the 3B weights |
| `ldi chat` | Talk with Granite 4.0 1B. Exit with `/quit` |
| `ldi ui` | Open a chat page on this Mac for models already on disk |
| `ldi ask "<question>"` | Answer from the passages with the local model |

Put your own files in `documents/` and run `ldi ingest documents` again. Unchanged files are skipped.

## Examples tested

The documents these answers were tested on are in this repo, under [examples/](examples/):

| Folder | What it holds |
|---|---|
| [examples/task1-original-public-documents](examples/task1-original-public-documents) | Public government and university documents: dearness-allowance circulars, the DAE allowances order, IIT Kanpur statutes, leave rules and forms, and a travel-claim workbook. |
| [examples/task2-invoice-ledger-corpus](examples/task2-invoice-ledger-corpus) | SKIT Jaipur library and e-governance invoices with their ledgers, plus `gold.csv`. Most pages are scans, so this is the hard extraction case. See its README. |
| [examples/task3-contract-version-corpus](examples/task3-contract-version-corpus) | One real version pair, EUPL 1.1 to EUPL 1.2, with `gold.csv` marking what changed and what stayed. See its README. |

Each corpus keeps its own README and sources file describing where the documents came from. They are public documents included here as test material.

## What was submitted

This is the source of truth for the TechKriti '27 / Lenovo AI PC Hackathon submission, Private AI Workbench, Problem C. The work covers three tasks: cited answers over HR policy, invoice and ledger reconciliation, and review of changes between contract versions. Every figure below is tied to the file it comes from. Where a prompt or a result file is not in this repo yet, that is said explicitly rather than reconstructed.

### The test documents

| Task | Documents | Location |
|---|---|---|
| Task 1, HR policy | Dearness-allowance circulars, the DAE allowances order, IIT Kanpur statutes, leave rules and forms | [examples/task1-original-public-documents](examples/task1-original-public-documents) |
| Task 2, invoices and ledgers | SKIT Jaipur library and e-governance invoices with their ledgers, and `gold.csv` | [examples/task2-invoice-ledger-corpus](examples/task2-invoice-ledger-corpus) |
| Task 3, contract versions | EUPL 1.1 and EUPL 1.2, with `gold.csv` marking changed and unchanged clauses | [examples/task3-contract-version-corpus](examples/task3-contract-version-corpus) |

### Model scores and latency by task

| Task and benchmark | Result | Azure CPU latency |
|---|---|---|
| 1 · Qwen2.5-3B 4-bit / Granite 4.0 1B 4-bit, paired choice | 43/44 / 32/44 placements | 49.5 / 24.2 s median per pair |
| 2 · Qwen3.5-0.8B / LFM3-1.7B, Q4_K_M with the same JSON constraint | 30/102 / 56/102 strict trials | 3.85 / 7.02 s median per request |
| 3 · Rules + keywords / with Gemma 3 270M, sealed legal edits | 70/80 / 75/80 changes flagged, plus two extra cosmetic checks (p = 0.063) | ≈0.66 s per pair for Gemma, as reported |

Task 1 timed 22 question pairs per model, each with both answer positions. Task 2 used prepared excerpts for 51 cases in two evidence orders. LFM3 returned valid JSON on 102/102 trials, copied 190/248 fields and missed all 14 absent-field checks. Azure accelerated benchmark iteration. The Workbench's Mac pipeline runs locally with a separate MLX/Metal runtime. [Run settings and source files](docs/submission-evidence.md#azure-model-benchmarks) · [Saved timing samples and benchmark counts](docs/codex-benchmark-evidence.json)

### Workbench evaluation

- **Policy Q&A.** Granite scored 7/26 on separate open questions. [Baseline counts](eval/baseline_counts.json)
- **Invoice OCR.** The Granite60 report records 29/33 exact matches for required invoice answer parts and 549 s to ingest 15 files with PaddleOCR-VL on the MacBook Air M4. [Mac and OCR measurement notes](docs/submission-evidence.md#mac-and-ocr-measurements)
- **Contract Q&A.** The Granite 4.0 1B Mac app passed 13/83 questions under strict scoring in v5 and v6. The Gemma experiment separately scored 299 held-out clause pairs against AI labels. Its 75/80 vs 70/80 sealed legal result fell short of the prespecified significance gate. [Contract reports and Gemma family breakdown](docs/submission-evidence.md#contract-qa-and-gemma)
- **Retrieval and verification.** The 1,200-character chunk sweep found gold evidence in the top 10 on 18/23 held-out questions. A 141-case repair replay and targeted fresh reruns passed 52/52 frozen regression and 30/30 adversarial checks. [Chunk sweep, product retrieval and replay cohorts](docs/submission-evidence.md#retrieval-and-replay)

### Weight-edit research

- **Granite 4.0 1B.** The last-layer MLP down-projection edit moved from 6/14 to 5/14. An independent edit of head 0 in layer 39’s attention output projection tied the original at 8/18 on fresh prompts. I kept the original checkpoint. [Saved results](docs/submission-evidence.md#granite-weight-edits) · [Exact development and evaluation prompts](docs/granite-weight-edit-prompts.json)
- **Qwen3.5-0.8B.** The layer 15/19 down-projection edit caused loops when applied at every generated token (0/20 on development). Applying it at the first token raised development passes from 1/20 to 10/20, with 0/19 correct abstentions after evidence removal. Both edits failed their guards, and I kept the original weights. [Weight-lab report and controls](docs/submission-evidence.md#qwen-weight-edit)

### Prompt for Task 1 (Qwen2.5-3B and Granite 4.0 1B)

Both models received the same prompt, wrapped in each model's own chat template. The choice probe appends the two options and asks for a single letter. The free-response probe drops the options and asks for one sentence. The exact text, from `eval/task1/run_baseline.py` in the benchmark lab:

```
Use only the supplied policy excerpts. Use the rate in effect on the date in the question. If the excerpts do not establish the answer, say so.

[file]
<excerpt>

Question: <question>

A. <option>
B. <option>
Reply with one letter only, A or B.
```

The free-response ending is `Answer in one short sentence.` The excerpts are the policy passages selected for that question.

### Prompt for Task 2 (Qwen3.5-0.8B and LFM3-1.7B)

Both models received the same extraction prompt, from `eval/task2/run_azure.py` in the benchmark lab. The JSON-constrained run adds a schema that fixes the array length and allows string or null items, plus the reminder `Use null for an absent field. Keep the value exactly as written.`

```
Extract the requested fields from the supplied ledger and document excerpts.
Copy the requested source field, preserving its role and units. Use null when the requested value is not directly shown in its source. A value on a different record is not evidence for that field. Do not calculate, infer, convert units, or answer in prose.
Return only one JSON array of exactly {count} items, in requested order. Each item is a string copied from the evidence or null. For example, ["example", null].

Evidence:
- <block>

Requested fields:
1. <field>
```

### Prompt for the Gemma 3 270M contract check

The probe reads one hidden state and generates no text. The prompt it was trained and scored on, filled per changed pair:

```
You compare two versions of one clause from a legal or policy document.
OLD: {old}
NEW: {new}
Is the difference only cosmetic (formatting, punctuation, capitalisation, renumbering, same meaning), or may the meaning have changed (rights, duties, scope, parties, amounts, dates, conditions)?
Answer with one word: Cosmetic or Meaning.
```

It is wrapped in Gemma's chat turn format as token ids: `[<bos>, <start_of_turn>]`, then `user` and the prompt, then `<end_of_turn>`, then `model`. The score combines the 640-value last-token hidden state at layer 9 and two rule features in a logistic regression. [Probe configuration and required assets](gemma_flagger/assets/README.md)

## Optional: Gemma meaning check

Version comparison already flags a clause that was replaced. This adds a second, optional check that marks a changed clause whose meaning may have changed, using Google's Gemma 3 270M (instruction-tuned) and a small probe trained on legal and policy text. It only ever adds a "check this: meaning may have changed" badge. It never removes or downgrades a flag the rules already set, it never rewrites a passage, and it never touches a citation.

The weights are not in this repo. Gemma is released under the Gemma Terms of Use, which is not the Apache licence, and this project does not redistribute the weights. You download them yourself:

1. Clone the repo and install the extra libraries:

```bash
uv sync
```

That pulls in `torch`, `transformers`, `sentencepiece`, and `huggingface_hub`.

2. Accept the Gemma Terms of Use on the model page at https://huggingface.co/google/gemma-3-270m-it and log in once with `huggingface-cli login`.
3. Download the weights:

```bash
python scripts/download_gemma.py
```

The download is checked against a known sha256 and stored under `models/`, which is gitignored. `GEMMA_DIR` overrides that path. If the gated repo is not an option, `python scripts/download_gemma.py --mirror` uses the same weights from `unsloth/gemma-3-270m-it`.

4. Place the separate frozen probe bundle in `gemma_flagger/assets/`, following its README. The model download supplies the checkpoint. The probe bundle supplies the classifier, configuration and reference rules. With both installed, turn on "Use Gemma meaning check" in the settings on the `ldi ui` page.

Two limits to keep in mind. The numbers reported for this check so far measure agreement with AI labels, not accuracy against a human-checked set. And it is tuned for legal and policy text: on other document types it adds a false check about 14% of the time.
