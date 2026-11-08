# whiscribe

Voice dictation for Linux. Records audio from any input device, transcribes it with [whisper.cpp](https://github.com/ggerganov/whisper.cpp), and copies the transcript text to the clipboard. Optionally saves to a file. Usable as a command-line tool (`whiscribe.py`) or a system tray app with a global shortcut (`tray.py`).

Targets KDE Plasma / Wayland on Manjaro Linux, but should work on any PipeWire/PulseAudio system with Wayland.

## Dependencies

### System packages

| Tool | Package (Arch/Manjaro) | Purpose |
|------|------------------------|---------|
| `whisper-cli` | `whisper.cpp` (AUR) | transcription engine |
| `arecord` | `alsa-utils` | audio recording |
| `pactl` | `libpulse` | device enumeration + BT profile switching |
| `wl-copy` | `wl-clipboard` | Wayland clipboard (optional; falls back to KDE Klipper via `qdbus`) |
| `vulkaninfo` | `vulkan-tools` | GPU detection for the tray label and Settings |
| `stdbuf` | `coreutils` | live transcription output (optional) |

### Python

Python 3.11 or newer (the config reader uses the stdlib `tomllib` module). The CLI
uses only standard-library modules. The optional tray app additionally needs
**PySide6** (`pip install pyside6`, or `pacman -S pyside6`).

### GPU acceleration (optional)

Vulkan drivers are required for GPU inference. On Intel/AMD integrated graphics:

```
sudo pacman -S vulkan-intel   # Intel
sudo pacman -S vulkan-radeon  # AMD
```

NVIDIA users need the proprietary driver with CUDA support compiled into whisper.cpp, which is beyond the scope of this README.

CPU-only mode works out of the box with no additional setup.

## Models

Model weights are not included in this repository. Download them from the
[whisper.cpp HuggingFace repo](https://huggingface.co/ggerganov/whisper.cpp) and place them in the `llm_models/` directory.

| Model | Size | Notes |
|-------|------|-------|
| `ggml-tiny.bin` | 75 MB | fastest, lowest accuracy |
| `ggml-base.bin` | 142 MB | |
| `ggml-small.bin` | 466 MB | good CPU trade-off |
| `ggml-medium.bin` | 1.5 GB | |
| `ggml-large-v3.bin` | 3.1 GB | best accuracy |
| `ggml-large-v3-turbo.bin` | 1.6 GB | near-large accuracy, much faster |

Download from the command line, or from **Settings…** in the tray menu:

```bash
./whiscribe.py --list-models            # what's available / present
./whiscribe.py --download-model ggml-small.bin
```

### Voice Activity Detection (optional, recommended)

A Silero VAD model lets whiscribe strip non-speech regions before transcription,
which sharply reduces hallucinated text on silence. It is enabled automatically
when the model file is present in `llm_models/`; disable per-run with `--no-vad`.

```bash
./whiscribe.py --download-model vad
```

## Installation

Make the script executable and optionally put it on your PATH:

```bash
chmod +x whiscribe.py

# Option A — symlink to ~/.local/bin (available system-wide)
ln -s "$(pwd)/whiscribe.py" ~/.local/bin/whiscribe

# Option B — add repo directory to PATH (in ~/.bashrc or ~/.zshrc)
export PATH="$HOME/development/whiscribe:$PATH"
```

## Usage

```
whiscribe.py [options]
```

On launch an interactive device picker lets you choose the input device with arrow keys.
Press **Ctrl+C** to stop recording. Transcription starts automatically.

### Configuration

On first run whiscribe writes a commented config file to
`~/.config/whiscribe/config.toml` (respecting `XDG_CONFIG_HOME`). Edit it to set your
defaults — most importantly the model, given by filename and looked up in `llm_models/`:

```toml
model = "ggml-large-v3.bin"   # filename inside llm_models/
threads = 4
language = ""                 # "" = auto-detect
vad = true
timestamps = false

delete_on_start = true        # saved recordings: see "Saved recordings" below
auto_delete = true
auto_delete_hours = 24
warn_size_mb = 1024
gpu = "auto"                  # or "off" to force CPU
gpu_device = 0
```

If the configured model isn't in `llm_models/`, whiscribe exits with an error listing
the models it did find. Command-line flags below override the config per run.

### Options

| Flag | Default | Description |
|------|---------|-------------|
| `-m, --model NAME` | from config | override the config model (bare name resolves in `llm_models/`) |
| `-o, --output FILE` | — | save transcript to this file (default: clipboard only) |
| `-t, --threads N` | from config | inference thread count |
| `-l, --language LANG` | from config | language hint, e.g. `en`, `el` |
| `--timestamps / --no-timestamps` | from config | include whisper timestamps in output |
| `--clip` | off | also copy to clipboard when `-o` is given |
| `--vad-model FILE` | `llm_models/ggml-silero-v5.1.2.bin` | Silero VAD model path |
| `--vad / --no-vad` | from config | strip non-speech regions before transcription |
| `--list-models` | — | show downloadable models and whether each is present, then exit |
| `--download-model NAME` | — | download a model into `llm_models/` (`vad` for the VAD model), then exit |

GPU inference via Vulkan is tried automatically first; falls back to CPU if it
fails. whiscribe also reads whisper's startup log, because a build or driver without
a usable GPU exits successfully and silently runs on the CPU — that case is reported
as a fallback too. The CLI prints the compute target before transcribing; `--cpu`
forces CPU for one run and `--gpu-device N` picks a GPU. Config keys `gpu`
(`"auto"` / `"off"`) and `gpu_device` set the defaults; the tray shows a
**Compute:** line in its menu, lets you change both in **Settings…**, and mentions
"GPU failed — used CPU" in the result notification if a fallback happened.

Clipboard uses `wl-copy` when available, falling back to KDE Klipper over D-Bus.

Recording is captured at 16 kHz mono (Whisper's native format). When the VAD model
is present, non-speech regions are stripped before inference (with padded speech
segments so words aren't clipped). Runaway repetition loops are prevented by disabling
cross-segment context carry, and any residual consecutive-duplicate lines are collapsed —
together the main defenses against hallucinated output.

### Examples

```bash
# Default — transcript text copied to clipboard (audio kept in recordings/)
whiscribe.py

# Save transcript to a file (clipboard not used)
whiscribe.py -o ~/notes/meeting.txt

# Save to file AND copy to clipboard
whiscribe.py -o ~/notes/meeting.txt --clip

# Keep whisper timestamps in the output (useful for LLM context or video work)
whiscribe.py -o ~/notes/meeting.txt --timestamps
```

## Output

The transcript text is copied to the clipboard and never stored by default. Use `-o FILE` to save it to a specific path instead. The audio is a different matter: every recording's WAV is kept for a while so a lost transcript can be redone — see [Saved recordings](#saved-recordings).

## System tray app

`tray.py` is a PySide6 tray front end sharing the same backend (device handling,
VAD, hallucination defenses, config). It reads the same `config.toml`; pick the
microphone from its menu (remembered across restarts), change the model and other
settings in the **Settings…** window (it can download missing models), or edit the
file via **Edit config…**.

### Install

Run the installer — it symlinks `whiscribe` and `whiscribe-tray` onto your PATH,
installs the app icon, adds a **Multimedia** menu entry, and sets up a systemd user
service that starts the tray at login and logs to the journal:

```bash
./install.sh
systemctl --user start whiscribe-tray     # start now (also autostarts on login)
```

Useful commands:

```bash
journalctl --user -u whiscribe-tray -e     # view logs / crashes
systemctl --user restart whiscribe-tray    # after pulling changes
systemctl --user disable --now whiscribe-tray   # stop autostarting
```

### Usage

Log in (the service autostarts it) or run `whiscribe-tray`. Left-click the tray icon
or use the menu:

- **Record to clipboard** / **Record to file…** — start recording; **Stop recording** ends it and transcribes.
- **Cancel** — abort, whether recording or transcribing (kills whisper, produces no text). The audio stays in **Previous transcriptions**.
- **Previous transcriptions** — saved recordings by timestamp: re-transcribe to the clipboard, play, or delete; **Clear transcriptions…** deletes all.
- **Settings…** — model (with download), threads, language, VAD, timestamps and the recording-retention options.
- Left-clicking the icon toggles recording to the clipboard.

### Global shortcuts (KDE Wayland)

The tray app can't grab a global hotkey directly on Wayland; instead bind keys to
commands that message the running instance over a local socket:

1. **System Settings → Shortcuts → Add → Command or Script**
2. Add `whiscribe-tray --toggle` and assign a key (e.g. `Ctrl+Meta+R`) — start/stop.
3. Add `whiscribe-tray --cancel` and assign another (e.g. `Ctrl+Meta+X`) — abort/discard.

(`Meta` is the Super/Win key.) Only one instance runs at a time; a second launch exits.

## Saved recordings

Every recording's audio (not the text) is kept in `recordings/` next to the
program, named by start time, whether it finished or was cancelled. In the tray,
**Previous transcriptions** lists them; each can be re-transcribed to the
clipboard, played, or deleted, and **Clear transcriptions…** removes all of them.
Audio is unencrypted speech on disk, so cleanup is on by default. Settings (all
editable in **Settings…** or `config.toml`):

| Key | Default | Meaning |
|-----|---------|---------|
| `delete_on_start` | `true` | delete all recordings when the tray starts |
| `auto_delete` | `true` | delete recordings older than the limit (hourly and before each recording) |
| `auto_delete_hours` | `24` | age limit in hours |
| `warn_size_mb` | `1024` | warn when saved recordings exceed this size |

Recording uses about 1.9 MB per minute, so 1 GB is roughly 9 hours of audio.
