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
