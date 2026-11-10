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

4. Turn on "Use Gemma meaning check" in the settings on the `ldi ui` page. The toggle stays off, and the rest of the app works as before, until the weights are on disk.

Two limits to keep in mind. The numbers reported for this check so far measure agreement with AI labels, not accuracy against a human-checked set. And it is tuned for legal and policy text: on other document types it adds a false check about 14% of the time.
