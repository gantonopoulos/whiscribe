# whiscribe

Voice dictation for Linux. Records audio from any input device, transcribes it with [whisper.cpp](https://github.com/ggerganov/whisper.cpp), and copies the result to the clipboard.

Targets KDE Plasma / Wayland on Manjaro Linux, but should work on any PipeWire/PulseAudio system with Wayland.

## Dependencies

### System packages

| Tool | Package (Arch/Manjaro) | Purpose |
|------|------------------------|---------|
| `whisper-cli` | `whisper.cpp` (AUR) | transcription engine |
| `arecord` | `alsa-utils` | audio recording |
| `pactl` | `libpulse` | device enumeration + BT profile switching |
| `wl-copy` | `wl-clipboard` | Wayland clipboard |
| `stdbuf` | `coreutils` | live transcription output (optional) |

### Python

Python 3.10 or newer. No pip dependencies — only standard library modules are used.

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

### Options

| Flag | Default | Description |
|------|---------|-------------|
| `-m, --model FILE` | `llm_models/ggml-small.bin` | whisper model to use |
| `-o, --output-dir DIR` | current directory | where to save output files |
| `-n, --filename NAME` | `whiscribe_YYYYMMDD_HHMMSS` | base name for `.wav` and `.txt` files |
| `-t, --threads N` | `4` | inference thread count |
| `--gpu` | off | enable GPU inference via Vulkan |
| `-l, --language LANG` | auto-detect | language hint, e.g. `en`, `el` |
| `--plain-text` | off | strip timestamps from saved text |
| `--clip` | `path` | clipboard content: `path` = `.txt` file path, `content` = transcript text |

### Examples

```bash
# Default — small model, CPU, copies file path to clipboard
whiscribe.py

# Large model with GPU, copy transcript text to clipboard
whiscribe.py -m llm_models/ggml-large-v3.bin --gpu --clip content

# Save to a specific directory with a fixed filename
whiscribe.py -o ~/notes -n meeting
```

## Output

Each recording produces two files:

- `whiscribe_YYYYMMDD_HHMMSS.wav` — raw audio
- `whiscribe_YYYYMMDD_HHMMSS.txt` — transcript (with timestamps by default)

By default the path to the `.txt` file is copied to the clipboard so it can be pasted directly as a file reference.
