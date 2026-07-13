# whiscribe

Voice dictation for Linux. Records audio from any input device, transcribes it with [whisper.cpp](https://github.com/ggerganov/whisper.cpp), and copies the transcript text to the clipboard. Optionally saves to a file.

Targets KDE Plasma / Wayland on Manjaro Linux, but should work on any PipeWire/PulseAudio system with Wayland.

## Dependencies

### System packages

| Tool | Package (Arch/Manjaro) | Purpose |
|------|------------------------|---------|
| `whisper-cli` | `whisper.cpp` (AUR) | transcription engine |
| `arecord` | `alsa-utils` | audio recording |
| `pactl` | `libpulse` | device enumeration + BT profile switching |
| `wl-copy` | `wl-clipboard` | Wayland clipboard (optional; falls back to KDE Klipper via `qdbus`) |
| `stdbuf` | `coreutils` | live transcription output (optional) |

### Python

Python 3.11 or newer (the config reader uses the stdlib `tomllib` module). No pip
dependencies — only standard library modules are used.

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

Quick download example (small model):

```bash
curl -L "https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-small.bin" \
     -o llm_models/ggml-small.bin
```

### Voice Activity Detection (optional, recommended)

A Silero VAD model lets whiscribe strip non-speech regions before transcription,
which sharply reduces hallucinated text on silence. It is enabled automatically
when the model file is present in `llm_models/`; disable per-run with `--no-vad`.

```bash
curl -L "https://huggingface.co/ggml-org/whisper-vad/resolve/main/ggml-silero-v5.1.2.bin" \
     -o llm_models/ggml-silero-v5.1.2.bin
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

GPU inference via Vulkan is tried automatically first; falls back to CPU if unavailable.

Clipboard uses `wl-copy` when available, falling back to KDE Klipper over D-Bus.

Recording is captured at 16 kHz mono (Whisper's native format). When the VAD model
is present, non-speech regions are stripped before inference (with padded speech
segments so words aren't clipped). Runaway repetition loops are prevented by disabling
cross-segment context carry, and any residual consecutive-duplicate lines are collapsed —
together the main defenses against hallucinated output.

### Examples

```bash
# Default — transcript text copied to clipboard, nothing saved to disk
whiscribe.py

# Save transcript to a file (clipboard not used)
whiscribe.py -o ~/notes/meeting.txt

# Save to file AND copy to clipboard
whiscribe.py -o ~/notes/meeting.txt --clip

# Keep whisper timestamps in the output (useful for LLM context or video work)
whiscribe.py -o ~/notes/meeting.txt --timestamps
```

## Output

By default nothing is saved to disk — the transcript text is copied directly to the clipboard. Use `-o FILE` to save to a specific path. The raw WAV recording is always discarded after transcription.
