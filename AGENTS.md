# Local document intelligence

This Mac does the compute. Document text, questions, and the index stay here.

## This Mac

| | |
|---|---|
| Machine | MacBook Air, Apple M4, fanless |
| CPU | 10 cores (4 performance + 6 efficiency) |
| GPU | 8 cores, Metal 4 |
| Memory | 16 GB unified, shared by the CPU, GPU, and Neural Engine |
| Storage | 256 GB SSD. Keep at least about 25 GB free. |
| macOS | 26.6.2 |

A local model will not damage this laptop. The M4 slows itself when it gets warm. That warmth on a long run is the chip protecting itself. macOS will stall or kill the model process before the hardware is at risk. A hang is recovered by quitting the model or restarting.

The real limits are memory and disk:

- With only Cursor open, plan on about 6–8 GB for the model.
- Keep at least about 25 GB free.
- Plug in for a long run. On battery it only drains faster.

When this was first measured, the Mac had been awake for 20 days, memory compression was heavy, and swap was about 8.3 GB. Free space was about 39 GB. A later check saw about 51 GB available and yellow memory pressure, with swap still in use. Trust `ldi status`, not these old numbers.

## What fits in 16 GB

| Model, 4-bit | On this Mac |
|---|---|
| 1B–3B | Easy first test. About 1–2 GB to download. |
| 7B–8B | The size to build the agent on. Weights about 4–5 GB. |
| 14B | Only with other apps quit. Heavier, uses compression. This project does not download it. |
| 32B and larger | Too big. Skip these. |

Short chats are cheap. A huge prompt uses memory beside the weights. This project uses MLX, SQLite full text, and a short passage list so only one model is loaded.

Safe order: restart once to clear a long-running session's swap, plug in, start with the 3B model, then the 7B model, and watch `ldi status`. Green or yellow memory pressure is fine. Red means the model stays unloaded.

## Cursor

One window, indexing paused, other apps quit. That, plus an 8B-class 4-bit model, plugged in, is a normal workload for this M4 Air. It will get warm.

## What the code does about this

- `ldi search` never loads a model. Text ingest never loads a model. A scanned page runs PaddleOCR-VL only while the chat model is unloaded, and that process exits before a chat model loads.
- `ldi setup` is the only networked command. It downloads one allowlisted repo. `ldi setup --model ocr` also creates the separate OCR environment.
- `ldi ask` answers from indexed passages and sets the hub offline before load.
- Allowed repos: `mlx-community/granite-4.0-1b-4bit`, `mlx-community/bge-small-en-v1.5-8bit`, `mlx-community/Qwen2.5-3B-Instruct-4bit`, `mlx-community/Qwen2.5-7B-Instruct-4bit`, and `jmbarrancoidener/PaddleOCR-VL-1.6-mlx-8bit`.
- Metal is capped at 512 MB for the embedding, 1280 MB for the 1B model from its measured peak, 2 GB for the OCR model, 4 GB for 3B, and 7 GB for 7B. The key-value cache is capped. The embedding unloads before a chat model loads. The OCR model loads only after the chat model unloads.
