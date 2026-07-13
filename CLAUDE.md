# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

`whiscribe.py` is a single-file, stdlib-only Python 3.10+ voice dictation tool for Linux. There is no build step, no package, and no third-party Python dependencies — it orchestrates external CLI binaries via `subprocess`.

## Running

```bash
./whiscribe.py                          # interactive device picker → record → transcribe → clipboard
./whiscribe.py -o notes.txt --clip      # save to file AND copy to clipboard
./whiscribe.py -m llm_models/ggml-base.bin -l en
```

There is no test suite or linter configured. Verifying changes requires a live machine with the runtime binaries and a working audio input, so most changes can only be checked by reading + `python3 -c "import ast; ast.parse(open('whiscribe.py').read())"` for syntax, or by the user running the tool.

## External binary dependencies (all invoked via subprocess)

The program is essentially glue around these; behavior depends heavily on their exact output format:

- `pactl` (libpulse) — device/card enumeration and Bluetooth profile switching. Code parses its human-readable `list sources` / `list cards` text output with regex, so it is sensitive to PulseAudio/PipeWire output format.
- `arecord` (alsa-utils) — records WAV via the `pulse` device, targeted using the `PULSE_SOURCE` env var.
- `whisper-cli` (whisper.cpp) — transcription. GPU (Vulkan) is attempted first; `--no-gpu` retry on non-zero exit.
- `wl-copy` (wl-clipboard) — Wayland clipboard + primary selection.
- `stdbuf` (coreutils, optional) — wrapped around whisper-cli to force line-buffered output so segments stream live.

## Architecture / control flow

`main()` runs a linear pipeline: parse args → check model/output paths → `list_input_devices()` → `pick_device()` (curses) → `prepare_device()` → `record()` → `transcribe_with_gpu_fallback()` → optionally strip timestamps → write file and/or clipboard.

Key design points worth knowing before editing:

- **Bluetooth is the main source of complexity.** A BT headset in A2DP mode exposes no input source. `list_input_devices()` merges real sources with BT cards that are in a no-input profile but have an available headset/HFP profile (`_parse_bt_cards_needing_switch`), de-duplicating by MAC so a card isn't listed twice. `prepare_device()` switches the profile at record time and polls (`find_bt_source_after_switch`) for the new source to appear. The original profile is **always restored in a `finally`** around `record()`.
- **The WAV is always a temp file and always deleted** (`tempfile.mkstemp` → `unlink` in `finally`). `-o` controls only the transcript text output, never the audio.
- **Output model:** clipboard-first. With no `-o`, transcript goes to the clipboard. With `-o`, it goes to the file only, unless `--clip` is also passed. See `copy_to_clipboard` logic near the end of `main()`.
- **Timestamp handling:** whisper prints `[hh:mm:ss --> hh:mm:ss]` prefixes; `strip_timestamps()` regex-removes them unless `--timestamps` is set.
- **Transcription quality:** recording is 16 kHz mono (Whisper-native). Hallucination defense is VAD-based, not decode-threshold-based (tightening `--no-speech`/`--logprob` risks dropping real quiet speech). When `DEFAULT_VAD_MODEL` exists in `llm_models/`, `transcribe()` passes `--vad --vad-model` with widened `--vad-speech-pad-ms`/`--vad-min-silence-duration-ms` (module constants) so words aren't clipped; auto-enabled, opt-out via `--no-vad`. Repetition-loop defense is two-layer: `-mc 0` (no cross-segment context carry) stops runaway repeats forming, and `collapse_repeats()` flattens any run of consecutive identical lines (anywhere, not just trailing) as a backstop. `--suppress-nst` drops non-speech tokens.
- **Bluetooth MAC matching:** PipeWire names BT sources with colons (`bluez_input.04:52:...`) but cards with underscores (`bluez_card.04_52_...`). `_mac_key()` normalizes both to separator-free lowercase hex; all source↔card matching (dedup, post-switch source discovery) must go through it, or devices double-list and profile-switch waits time out.

## Configuration

`load_config()` reads `~/.config/whiscribe/config.toml` (XDG), writing a commented
template on first run. Config supplies argparse **defaults**; CLI flags override per run.
The `model` key is a filename resolved inside `llm_models/` by `resolve_model_path()`
(bare name → `llm_models/`; path with a separator → used as-is); a missing model exits
with an error listing `available_models()`. Reading uses stdlib `tomllib` (Python 3.11+);
there is no config writer, so the tray app (future) will need one or a separate store.

## Models

Model weights are not in the repo (gitignored under `llm_models/*.bin`). Default is `llm_models/ggml-small.bin`. See README for download links.
