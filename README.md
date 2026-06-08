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
| `-o, --output FILE` | — | save transcript to this file (default: clipboard only) |
| `-t, --threads N` | `4` | inference thread count |
| `-l, --language LANG` | auto-detect | language hint, e.g. `en`, `el` |
| `--timestamps` | off | include whisper timestamps in output |
| `--clip` | off | also copy to clipboard when `-o` is given |

GPU inference via Vulkan is tried automatically first; falls back to CPU if unavailable.

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
