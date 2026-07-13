# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

A Python 3.11+ voice dictation tool for Linux that orchestrates external CLI binaries via `subprocess`. Two front ends share one backend:

- **`backend.py`** — all logic (device discovery, Bluetooth, recording, transcription, config, clipboard). No UI; failures raise exceptions (not `sys.exit`) so the GUI can recover. Stdlib-only.
- **`cli.py`** — terminal front end: curses device picker + argparse `main()`. Translates backend exceptions to `sys.exit`.
- **`whiscribe.py`** — thin shim → `cli.main()` (keeps the PATH symlink working; inserts its resolved dir on `sys.path` so siblings import through a symlink).
- **`tray.py`** — PySide6 system tray app (the only component needing a third-party dep).

## Running

```bash
./whiscribe.py                          # CLI: device picker → record → transcribe → clipboard
./whiscribe.py -o notes.txt --clip      # save to file AND copy to clipboard
./whiscribe-tray                        # tray app (needs PySide6)
```

There is no test suite or linter. Verifying changes needs a live machine with the runtime binaries + audio input, so most checks are: `python3 -c "import ast; ast.parse(open('backend.py').read())"` for syntax, `python3 -c "import backend; print(backend.list_input_devices())"` for the device path, and the user running the tool. The tray launches on the live Wayland session; a second launch exits with "already running" (single-instance guard).

## External binary dependencies (all invoked via subprocess)

The program is essentially glue around these; behavior depends heavily on their exact output format:

- `pactl` (libpulse) — device/card enumeration and Bluetooth profile switching. Code parses its human-readable `list sources` / `list cards` text output with regex, so it is sensitive to PulseAudio/PipeWire output format.
- `arecord` (alsa-utils) — records WAV via the `pulse` device, targeted using the `PULSE_SOURCE` env var.
- `whisper-cli` (whisper.cpp) — transcription. GPU (Vulkan) is attempted first; `--no-gpu` retry on non-zero exit.
- `wl-copy` (wl-clipboard) — Wayland clipboard + primary selection.
- `stdbuf` (coreutils, optional) — wrapped around whisper-cli to force line-buffered output so segments stream live.

## Architecture / control flow

The CLI `main()` (in `cli.py`) runs a linear pipeline: parse args → check model/output paths → `list_input_devices()` → `pick_device()` (curses) → `prepare_device()` → `record()` → `transcribe_with_gpu_fallback()` → optionally strip timestamps → write file and/or clipboard. The tray (`tray.py`) drives the same backend calls but split across two `QThread`s (`RecordWorker`, `TranscribeWorker`) so the UI stays responsive; its signals are named `done`/`failed` (not `finished`, which would shadow `QThread.finished`).

Key design points worth knowing before editing:

- **Bluetooth is the main source of complexity.** A BT headset in A2DP mode exposes no input source. `list_input_devices()` merges real sources with BT cards that are in a no-input profile but have an available headset/HFP profile (`_parse_bt_cards_needing_switch`), de-duplicating by MAC so a card isn't listed twice. `prepare_device()` switches the profile at record time and polls (`find_bt_source_after_switch`) for the new source to appear. The original profile is **always restored in a `finally`** (around `record()` in the CLI; in `RecordWorker.run` for the tray).
- **Recording:** `Recorder` wraps `arecord`. `start()`+`wait()` is the blocking CLI path (Ctrl-C); `start()`+`stop()` (SIGINT) is the tray path, since the tray's `arecord` isn't in the terminal's process group. The WAV is always a temp file and always deleted; `-o` controls only the transcript text output, never the audio.
- **Output model:** clipboard-first. With no `-o`, transcript goes to the clipboard. With `-o`, it goes to the file only, unless `--clip` is also passed.
- **Tray IPC / global shortcut:** `tray.py` runs a `QLocalServer` named `whiscribe-tray` for single-instance + commands. `whiscribe-tray --toggle` / `--cancel` connect as a client and send `toggle`/`cancel`; KDE custom shortcuts bind keys to those commands (Wayland can't grab a hotkey in-process). Tray-only UI state (selected mic) persists in `tray_state.json`; transcription settings still come from the shared `config.toml` (read-only from the tray, edited via **Edit config…** → `xdg-open`).
- **Cancel semantics:** `cancel()` discards. Mid-recording it sets `_cancelled` and stops the recorder (which `_on_recorded`/`_on_error` honor by discarding + resetting). Mid-transcription it sets the worker's `threading.Event`, which `backend.transcribe` watches (a watcher thread `terminate()`s whisper-cli even during model load) and raises `backend.Cancelled` — `transcribe_with_gpu_fallback` lets it propagate so there is **no** CPU retry on cancel. Note `RecordWorker.stop()` sets a `_stop_requested` flag re-checked in `run()` after `start()`, since a fast toggle→cancel can call `stop()` before the `Recorder` exists.
- **Timestamp handling:** whisper prints `[hh:mm:ss --> hh:mm:ss]` prefixes; `strip_timestamps()` regex-removes them unless `--timestamps` is set.
- **Transcription quality:** recording is 16 kHz mono (Whisper-native). Hallucination defense is VAD-based, not decode-threshold-based (tightening `--no-speech`/`--logprob` risks dropping real quiet speech). When `DEFAULT_VAD_MODEL` exists in `llm_models/`, `transcribe()` passes `--vad --vad-model` with widened `--vad-speech-pad-ms`/`--vad-min-silence-duration-ms` (module constants) so words aren't clipped; auto-enabled, opt-out via `--no-vad`. Repetition-loop defense is two-layer: `-mc 0` (no cross-segment context carry) stops runaway repeats forming, and `collapse_repeats()` flattens any run of consecutive identical lines (anywhere, not just trailing) as a backstop. `--suppress-nst` drops non-speech tokens.
- **Bluetooth MAC matching:** PipeWire names BT sources with colons (`bluez_input.04:52:...`) but cards with underscores (`bluez_card.04_52_...`). `_mac_key()` normalizes both to separator-free lowercase hex; all source↔card matching (dedup, post-switch source discovery) must go through it, or devices double-list and profile-switch waits time out.

## Configuration

`load_config()` reads `~/.config/whiscribe/config.toml` (XDG), writing a commented
template on first run. Config supplies argparse **defaults**; CLI flags override per run.
The `model` key is a filename resolved inside `llm_models/` by `resolve_model_path()`
(bare name → `llm_models/`; path with a separator → used as-is); a missing model exits
with an error listing `available_models()`. Reading uses stdlib `tomllib` (Python 3.11+);
there is no config writer — the tray stores its own UI state (mic) in `tray_state.json`
and leaves `config.toml` for hand-editing.

## Models

Model weights are not in the repo (gitignored under `llm_models/*.bin`). The config default is `ggml-large-v3.bin`. See README for download links.
