#!/usr/bin/env python3
"""whiscribe — voice recording + transcription via whisper.cpp"""

import argparse
import curses
import os
import pathlib
import re
import shutil
import subprocess
import sys
import tempfile
import time
import tomllib
from typing import NamedTuple


SCRIPT_DIR = pathlib.Path(__file__).parent.resolve()

# Whisper models live here; the config file names one by filename.
MODELS_DIR = SCRIPT_DIR / "llm_models"

# Voice Activity Detection model. Auto-enabled when present; download from
# https://huggingface.co/ggml-org/whisper-vad and place in llm_models/.
DEFAULT_VAD_MODEL = MODELS_DIR / "ggml-silero-v5.1.2.bin"

# --- Configuration --------------------------------------------------------
# Hand-edited TOML at ~/.config/whiscribe/config.toml. CLI flags override it.
CONFIG_DIR = pathlib.Path(
    os.environ.get("XDG_CONFIG_HOME", pathlib.Path.home() / ".config")
) / "whiscribe"
CONFIG_PATH = CONFIG_DIR / "config.toml"

DEFAULT_CONFIG = {
    "model": "ggml-large-v3.bin",  # filename inside llm_models/
    "threads": 4,
    "language": "",                # "" = auto-detect
    "vad": True,
    "timestamps": False,
}

CONFIG_TEMPLATE = """\
# whiscribe configuration

# Whisper model filename, looked up in the llm_models/ directory next to
# whiscribe.py. Download models from
# https://huggingface.co/ggerganov/whisper.cpp
#   ggml-large-v3.bin        best accuracy (needs a capable GPU)
#   ggml-large-v3-turbo.bin  near-large accuracy, much faster
#   ggml-small.bin           lightweight CPU-friendly fallback
model = "ggml-large-v3.bin"

# Threads for whisper inference.
threads = 4

# Language hint, e.g. "en" or "el". Leave empty for auto-detect.
language = ""

# Strip non-speech regions before transcription (needs the Silero VAD model
# in llm_models/). Strongly recommended — reduces hallucinated output.
vad = true

# Include whisper timestamps in the output.
timestamps = false
"""

# VAD tuning passed to whisper-cli. whisper.cpp pads detected speech by only
# 30 ms, which clips word onsets/endings; widen it so no words are lost. Also
# require a longer silence before splitting so mid-sentence pauses don't cut
# words. VAD (not tightened decode thresholds) is our hallucination defense —
# tightening --no-speech/--logprob risks dropping real quiet speech.
VAD_SPEECH_PAD_MS = "200"
VAD_MIN_SILENCE_MS = "500"

# Native Whisper input format — recording here avoids a downsample step.
RECORD_RATE = "16000"
RECORD_CHANNELS = "1"


class InputDevice(NamedTuple):
    label: str
    source_name: str | None        # None = BT card needs profile switch first
    bt_card_name: str | None       # e.g. bluez_card.04_52_C7_79_02_3F
    bt_target_profile: str | None  # profile to switch to for recording
    bt_saved_profile: str | None   # profile to restore after recording


# ---------------------------------------------------------------------------
# Audio device discovery
# ---------------------------------------------------------------------------

def list_audio_sources() -> list[tuple[str, str]]:
    """Return (pulse_name, human_description) for every non-monitor input source."""
    try:
        result = subprocess.run(
            ["pactl", "list", "sources"],
            capture_output=True, text=True, check=True,
        )
    except FileNotFoundError:
        sys.exit("pactl not found — is PulseAudio/PipeWire running?")

    sources: list[tuple[str, str]] = []
    current_name: str | None = None

    for line in result.stdout.splitlines():
        stripped = line.strip()
        if stripped.startswith("Name:"):
            current_name = stripped.split(":", 1)[1].strip()
        elif stripped.startswith("Description:") and current_name:
            desc = stripped.split(":", 1)[1].strip()
            if ".monitor" not in current_name:
                sources.append((current_name, desc))
            current_name = None

    return sources


def _parse_bt_cards_needing_switch(output: str) -> list[dict]:
    """
    Parse 'pactl list cards' output and return BT cards that are currently in a
    no-input profile (e.g. A2DP) but have an available headset/HFP profile.
    """
    cards = []
    blocks = re.split(r"(?=^Card #)", output, flags=re.MULTILINE)

    for block in blocks:
        if "bluez_card" not in block:
            continue

        name_m = re.search(r"Name:\s*(bluez_card\.\S+)", block)
        if not name_m:
            continue
        name = name_m.group(1)

        desc_m = re.search(r'device\.(?:description|alias)\s*=\s*"([^"]+)"', block)
        description = desc_m.group(1) if desc_m else name

        active_m = re.search(r"Active Profile:\s*(\S+)", block)
        if not active_m:
            continue
        active_profile = active_m.group(1)

        # Skip if the current profile already exposes an input source
        active_src_m = re.search(
            rf"^\s+{re.escape(active_profile)}:.*?sources:\s*([0-9]+)",
            block, re.MULTILINE,
        )
        if active_src_m and int(active_src_m.group(1)) > 0:
            continue

        # Find available profiles that expose at least one source
        headset_profiles = re.findall(
            r"^\s+(\S*(?:headset|handsfree|hfp|hsp)\S*):\s.*?sources:\s*([1-9]).*?available:\s*yes",
            block, re.IGNORECASE | re.MULTILINE,
        )
        if not headset_profiles:
            continue

        cards.append({
            "name": name,
            "description": description,
            "active_profile": active_profile,
            "target_profile": headset_profiles[0][0],
        })

    return cards


def _mac_key(name: str) -> str | None:
    """Extract a Bluetooth MAC from a bluez source/card name as a normalized key
    (lowercase hex, no separators). PipeWire names sources with colons
    (bluez_input.04:52:C7:...) but cards with underscores (bluez_card.04_52_C7_...),
    so both must collapse to the same key to match a source with its card."""
    m = re.search(r"bluez[._][a-z]+\.([0-9A-Fa-f]+(?:[:_][0-9A-Fa-f]+)+)", name)
    if not m:
        return None
    return re.sub(r"[^0-9a-f]", "", m.group(1).lower())


def list_input_devices() -> list[InputDevice]:
    """Return all recording-capable devices, including BT cards currently in A2DP mode."""
    devices: list[InputDevice] = []
    bt_macs_covered: set[str] = set()

    for source_name, desc in list_audio_sources():
        devices.append(InputDevice(
            label=desc,
            source_name=source_name,
            bt_card_name=None,
            bt_target_profile=None,
            bt_saved_profile=None,
        ))
        mac = _mac_key(source_name)
        if mac:
            bt_macs_covered.add(mac)

    try:
        result = subprocess.run(
            ["pactl", "list", "cards"],
            capture_output=True, text=True, check=True,
        )
        for card in _parse_bt_cards_needing_switch(result.stdout):
            mac = _mac_key(card["name"])
            if mac and mac in bt_macs_covered:
                continue
            devices.append(InputDevice(
                label=f"{card['description']}  [will switch to headset mode]",
                source_name=None,
                bt_card_name=card["name"],
                bt_target_profile=card["target_profile"],
                bt_saved_profile=card["active_profile"],
            ))
    except subprocess.CalledProcessError:
        pass  # BT card scan is best-effort

    return devices


# ---------------------------------------------------------------------------
# Bluetooth helpers
# ---------------------------------------------------------------------------

def switch_bt_profile(card_name: str, profile: str) -> None:
    subprocess.run(["pactl", "set-card-profile", card_name, profile], check=True)


def find_bt_source_after_switch(bt_card_name: str, timeout: float = 3.0) -> str | None:
    """Poll until the BT card exposes an input source; return its name or None."""
    mac = _mac_key(bt_card_name)
    if not mac:
        return None

    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        for source_name, _ in list_audio_sources():
            if _mac_key(source_name) == mac:
                return source_name
        time.sleep(0.5)

    return None


# ---------------------------------------------------------------------------
# Interactive curses picker
# ---------------------------------------------------------------------------

def _run_picker(stdscr: curses.window, items: list[str]) -> int:
    curses.curs_set(0)
    curses.start_color()
    curses.init_pair(1, curses.COLOR_BLACK, curses.COLOR_WHITE)

    selected = 0
    while True:
        stdscr.erase()
        h, w = stdscr.getmaxyx()

        header = "Select input device  [↑/↓ or j/k · Enter to confirm · q to quit]"
        stdscr.addstr(0, 0, header[:w - 1], curses.A_BOLD)

        for i, item in enumerate(items):
            y = i + 2
            if y >= h - 1:
                break
            label = f"  {item}"[:w - 1]
            attr = curses.color_pair(1) if i == selected else curses.A_NORMAL
            stdscr.addstr(y, 0, label, attr)

        stdscr.refresh()
        key = stdscr.getch()

        if key in (curses.KEY_UP, ord("k")) and selected > 0:
            selected -= 1
        elif key in (curses.KEY_DOWN, ord("j")) and selected < len(items) - 1:
            selected += 1
        elif key in (curses.KEY_ENTER, ord("\n"), ord("\r")):
            return selected
        elif key in (ord("q"), 27):
            return -1


def pick_device(devices: list[InputDevice]) -> InputDevice | None:
    labels = [d.label for d in devices]
    idx = curses.wrapper(_run_picker, labels)
    if idx < 0:
        return None
    chosen = devices[idx]
    print(f"Selected: {chosen.label}")
    return chosen


# ---------------------------------------------------------------------------
# Recording
# ---------------------------------------------------------------------------

def prepare_device(device: InputDevice) -> tuple[str, str | None, str | None]:
    """
    Ensure the device is ready for recording.
    For BT cards in A2DP mode: switch profile and wait for the input source to appear.
    Returns (source_name, bt_card_to_restore, profile_to_restore).
    """
    if device.source_name:
        return device.source_name, None, None

    print("Switching Bluetooth to headset mode...")
    switch_bt_profile(device.bt_card_name, device.bt_target_profile)
    source_name = find_bt_source_after_switch(device.bt_card_name)
    if not source_name:
        sys.exit("Timed out waiting for headset audio source after Bluetooth profile switch.")
    return source_name, device.bt_card_name, device.bt_saved_profile


def record(source_name: str, output_path: pathlib.Path) -> None:
    """Record from source_name into a WAV file; Ctrl-C stops cleanly."""
    print("\nRecording... Press Ctrl+C to stop.\n")
    env = {**os.environ, "PULSE_SOURCE": source_name}
    proc = subprocess.Popen(
        ["arecord", "-D", "pulse",
         "-f", "S16_LE", "-r", RECORD_RATE, "-c", RECORD_CHANNELS,
         "-t", "wav", str(output_path)],
        env=env,
        stderr=subprocess.DEVNULL,
    )
    try:
        proc.wait()
    except KeyboardInterrupt:
        # arecord received SIGINT from the terminal; wait for it to flush the WAV header
        proc.wait()


# ---------------------------------------------------------------------------
# Transcription
# ---------------------------------------------------------------------------

def transcribe(
    wav_path: pathlib.Path,
    model_path: pathlib.Path,
    threads: int,
    no_gpu: bool,
    language: str | None,
    vad_model: pathlib.Path | None = None,
) -> str | None:
    """Run whisper-cli, stream each transcript segment to the terminal, return full text or None on failure."""
    cmd = [
        "whisper-cli",
        "-m", str(model_path),
        "-f", str(wav_path),
        "-t", str(threads),
        # No cross-segment text context: Whisper repetition loops are fed by the
        # model conditioning on its own prior (already-repeating) output, so
        # zeroing the carried context is the structural cure for runaway repeats.
        "-mc", "0",
        # Suppress non-speech tokens (blank-audio / music markers).
        "--suppress-nst",
    ]
    # VAD strips non-speech regions before inference — the biggest single
    # reducer of hallucinated text on silence. Only usable if the model is present.
    if vad_model and vad_model.exists():
        cmd += [
            "--vad", "--vad-model", str(vad_model),
            "--vad-speech-pad-ms", VAD_SPEECH_PAD_MS,
            "--vad-min-silence-duration-ms", VAD_MIN_SILENCE_MS,
        ]
    if no_gpu:
        cmd.append("--no-gpu")
    if language:
        cmd += ["-l", language]

    # stdbuf forces line-buffered stdout so segments appear in real time over a pipe
    if shutil.which("stdbuf"):
        cmd = ["stdbuf", "-oL"] + cmd

    proc = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
        text=True,
    )
    lines: list[str] = []
    for raw_line in proc.stdout:
        line = raw_line.rstrip("\n")
        if line:
            print(line)
            lines.append(line)
    proc.wait()
    if proc.returncode != 0:
        return None

    return "\n".join(lines)


def transcribe_with_gpu_fallback(
    wav_path: pathlib.Path,
    model_path: pathlib.Path,
    threads: int,
    language: str | None,
    vad_model: pathlib.Path | None = None,
) -> str:
    """Try GPU transcription first; fall back to CPU if it fails."""
    print("\nTranscribing (GPU)...")
    result = transcribe(wav_path, model_path, threads, no_gpu=False, language=language, vad_model=vad_model)
    if result is not None:
        return result

    print("GPU transcription failed, retrying on CPU...")
    result = transcribe(wav_path, model_path, threads, no_gpu=True, language=language, vad_model=vad_model)
    if result is None:
        sys.exit("Transcription failed on both GPU and CPU.")
    return result


def strip_timestamps(text: str) -> str:
    return re.sub(r"\[[\d:.]+ --> [\d:.]+\]\s*", "", text).strip()


def collapse_repeats(text: str) -> str:
    """Collapse any run of consecutive identical lines to a single occurrence.
    Whisper repetition loops emit the same line many times (anywhere in the
    output, not just at the end); real dictation almost never repeats a full
    line verbatim back-to-back."""
    out: list[str] = []
    for line in text.split("\n"):
        if out and out[-1].strip() and line.strip() == out[-1].strip():
            continue
        out.append(line)
    return "\n".join(out)


def set_clipboard(text: str) -> bool:
    """Copy text to the clipboard. Prefers wl-copy (Wayland); falls back to KDE
    Klipper over D-Bus so it still works when wl-clipboard isn't installed.
    Returns True if some method succeeded."""
    if shutil.which("wl-copy"):
        subprocess.run(["wl-copy"], input=text, text=True, check=False)
        subprocess.run(["wl-copy", "--primary"], input=text, text=True, check=False)
        return True

    for qdbus in ("qdbus6", "qdbus"):
        if shutil.which(qdbus):
            result = subprocess.run(
                [qdbus, "org.kde.klipper", "/klipper", "setClipboardContents", text],
                capture_output=True, text=True,
            )
            if result.returncode == 0:
                return True

    return False


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

def load_config() -> dict:
    """Load ~/.config/whiscribe/config.toml, creating it from a template on first
    run. Unknown/missing keys fall back to DEFAULT_CONFIG."""
    cfg = dict(DEFAULT_CONFIG)

    if not CONFIG_PATH.exists():
        CONFIG_DIR.mkdir(parents=True, exist_ok=True)
        CONFIG_PATH.write_text(CONFIG_TEMPLATE, encoding="utf-8")
        print(f"Created default config: {CONFIG_PATH}")

    try:
        with open(CONFIG_PATH, "rb") as f:
            data = tomllib.load(f)
    except (OSError, tomllib.TOMLDecodeError) as exc:
        print(f"Config error in {CONFIG_PATH}: {exc}\nUsing built-in defaults.")
        return cfg

    for key in cfg:
        if key in data:
            cfg[key] = data[key]
    return cfg


def available_models() -> list[str]:
    """Whisper .bin models present in llm_models/ (excludes the VAD model)."""
    if not MODELS_DIR.is_dir():
        return []
    return sorted(
        p.name for p in MODELS_DIR.glob("*.bin") if "silero" not in p.name.lower()
    )


def resolve_model_path(name: str) -> pathlib.Path:
    """Resolve a model name to a path. Bare filenames resolve inside llm_models/;
    paths (absolute or containing a separator) are used as given."""
    candidate = pathlib.Path(name).expanduser()
    if candidate.is_absolute() or os.sep in name:
        return candidate
    return MODELS_DIR / name


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main() -> None:
    cfg = load_config()
    parser = argparse.ArgumentParser(
        description="Record audio and transcribe with whisper.cpp",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "-m", "--model", metavar="NAME", default=None,
        help=f"Override the config model (default from config: {cfg['model']}). "
             "Bare names resolve inside llm_models/.",
    )
    parser.add_argument(
        "-o", "--output", type=pathlib.Path,
        metavar="FILE",
        help="Save transcript to this file path (default: clipboard only, no file saved)",
    )
    parser.add_argument(
        "-t", "--threads", type=int, default=cfg["threads"],
        help="Number of threads for whisper inference",
    )
    parser.add_argument(
        "-l", "--language", default=cfg["language"] or None,
        metavar="LANG",
        help="Language code hint for whisper (e.g. 'en', 'el'); auto-detect if omitted",
    )
    parser.add_argument(
        "--timestamps", action=argparse.BooleanOptionalAction, default=cfg["timestamps"],
        help="Include whisper timestamps in output",
    )
    parser.add_argument(
        "--clip", action="store_true",
        help="Also copy transcript to clipboard when -o is given (default with -o: file only)",
    )
    parser.add_argument(
        "--vad-model", type=pathlib.Path, default=DEFAULT_VAD_MODEL,
        metavar="FILE",
        help="Silero VAD model; enables speech-region filtering when the file exists",
    )
    parser.add_argument(
        "--vad", action=argparse.BooleanOptionalAction, default=cfg["vad"],
        help="Strip non-speech regions before transcription (needs the VAD model)",
    )

    args = parser.parse_args()

    # --- resolve the model: CLI overrides config; must exist in llm_models/ ---
    model_path = resolve_model_path(args.model or cfg["model"])
    if not model_path.exists():
        have = available_models()
        listing = "\n  ".join(have) if have else "(none)"
        sys.exit(
            f"Model not found: {model_path}\n"
            f"Set 'model' in {CONFIG_PATH} to one of the models in {MODELS_DIR}:\n"
            f"  {listing}"
        )

    vad_model = args.vad_model if args.vad else None
    if vad_model and not vad_model.exists():
        print(f"Note: VAD model not found at {vad_model} — transcribing without VAD.")

    # --- check output file before doing anything ---
    if args.output and args.output.exists():
        answer = input(f"File already exists: {args.output}\nOverwrite? [y/N] ").strip().lower()
        if answer != "y":
            print("Aborted.")
            sys.exit(0)

    # --- device selection ---
    devices = list_input_devices()
    if not devices:
        sys.exit("No audio input devices found.")

    device = pick_device(devices)
    if device is None:
        print("Aborted.")
        sys.exit(0)

    # --- prepare device (switches BT profile if needed) ---
    source_name, bt_card, bt_saved_profile = prepare_device(device)

    # --- WAV goes to a temp file, always deleted after transcription ---
    tmp_fd, tmp_wav = tempfile.mkstemp(suffix=".wav", prefix="whiscribe_")
    os.close(tmp_fd)
    wav_path = pathlib.Path(tmp_wav)

    # --- record; always restore BT profile in finally ---
    try:
        record(source_name, wav_path)
    finally:
        if bt_card and bt_saved_profile:
            print("Restoring Bluetooth profile...")
            switch_bt_profile(bt_card, bt_saved_profile)

    try:
        if not wav_path.exists() or wav_path.stat().st_size < 1024:
            sys.exit("No usable recording — WAV file is missing or too small.")

        # --- transcribe (GPU first, CPU fallback) ---
        raw_text = transcribe_with_gpu_fallback(
            wav_path, model_path, args.threads, args.language, vad_model=vad_model)
        print()
    finally:
        wav_path.unlink(missing_ok=True)

    if not raw_text:
        print("Warning: whisper returned empty output.")

    final_text = raw_text if args.timestamps else strip_timestamps(raw_text)
    final_text = collapse_repeats(final_text)

    # --- save to file if -o given ---
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(final_text + "\n", encoding="utf-8")
        print(f"Saved: {args.output}")

    # --- clipboard: always in default mode; only with --clip when -o is given ---
    copy_to_clipboard = (args.output is None) or args.clip
    if copy_to_clipboard:
        if set_clipboard(final_text):
            print("Copied to clipboard.")
        else:
            print("Clipboard not updated — install wl-clipboard (or start KDE Klipper).")


if __name__ == "__main__":
    main()
