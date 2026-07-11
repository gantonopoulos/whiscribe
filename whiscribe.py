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
from typing import NamedTuple


SCRIPT_DIR = pathlib.Path(__file__).parent.resolve()
DEFAULT_MODEL = SCRIPT_DIR / "llm_models" / "ggml-small.bin"

# Voice Activity Detection model. Auto-enabled when present; download from
# https://huggingface.co/ggml-org/whisper-vad and place in llm_models/.
DEFAULT_VAD_MODEL = SCRIPT_DIR / "llm_models" / "ggml-silero-v5.1.2.bin"

# Quality tuning passed to whisper-cli. Tighter than whisper.cpp defaults
# (no-speech 0.60, logprob -1.00) to reject low-confidence / silent segments
# that Whisper otherwise hallucinates text into.
NO_SPEECH_THRESHOLD = "0.45"
LOGPROB_THRESHOLD = "-0.7"

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
        mac_m = re.search(r"bluez[._][a-z]+\.([0-9A-Fa-f_]+)", source_name, re.IGNORECASE)
        if mac_m:
            bt_macs_covered.add(mac_m.group(1))

    try:
        result = subprocess.run(
            ["pactl", "list", "cards"],
            capture_output=True, text=True, check=True,
        )
        for card in _parse_bt_cards_needing_switch(result.stdout):
            mac_m = re.search(r"bluez_card\.([0-9A-Fa-f_]+)", card["name"], re.IGNORECASE)
            mac = mac_m.group(1) if mac_m else None
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
    mac_m = re.search(r"bluez_card\.([0-9A-Fa-f_]+)", bt_card_name, re.IGNORECASE)
    if not mac_m:
        return None
    mac = mac_m.group(1).lower()

    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        for source_name, _ in list_audio_sources():
            if mac in source_name.lower():
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
        "--no-speech-thold", NO_SPEECH_THRESHOLD,
        "--logprob-thold", LOGPROB_THRESHOLD,
    ]
    # VAD strips non-speech regions before inference — the biggest single
    # reducer of hallucinated text on silence. Only usable if the model is present.
    if vad_model and vad_model.exists():
        cmd += ["--vad", "--vad-model", str(vad_model)]
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


def collapse_trailing_repeats(text: str) -> str:
    """Drop consecutive identical trailing lines — a common Whisper hallucination
    loop (e.g. repeated 'Thank you.') on trailing near-silence."""
    lines = text.split("\n")
    while len(lines) >= 2 and lines[-1].strip() and lines[-1].strip() == lines[-2].strip():
        lines.pop()
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Record audio and transcribe with whisper.cpp",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "-m", "--model", type=pathlib.Path, default=DEFAULT_MODEL,
        metavar="FILE",
        help="Path to whisper .bin model file",
    )
    parser.add_argument(
        "-o", "--output", type=pathlib.Path,
        metavar="FILE",
        help="Save transcript to this file path (default: clipboard only, no file saved)",
    )
    parser.add_argument(
        "-t", "--threads", type=int, default=4,
        help="Number of threads for whisper inference",
    )
    parser.add_argument(
        "-l", "--language",
        metavar="LANG",
        help="Language code hint for whisper (e.g. 'en', 'el'); auto-detect if omitted",
    )
    parser.add_argument(
        "--timestamps", action="store_true",
        help="Include whisper timestamps in output (default: plain text)",
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
        "--no-vad", action="store_true",
        help="Disable Voice Activity Detection even if a VAD model is present",
    )

    args = parser.parse_args()

    if not args.model.exists():
        sys.exit(f"Model not found: {args.model}")

    vad_model = None if args.no_vad else args.vad_model
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
            wav_path, args.model, args.threads, args.language, vad_model=vad_model)
        print()
    finally:
        wav_path.unlink(missing_ok=True)

    if not raw_text:
        print("Warning: whisper returned empty output.")

    final_text = raw_text if args.timestamps else strip_timestamps(raw_text)
    final_text = collapse_trailing_repeats(final_text)

    # --- save to file if -o given ---
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(final_text + "\n", encoding="utf-8")
        print(f"Saved: {args.output}")

    # --- clipboard: always in default mode; only with --clip when -o is given ---
    copy_to_clipboard = (args.output is None) or args.clip
    if copy_to_clipboard:
        if shutil.which("wl-copy"):
            subprocess.run(["wl-copy"], input=final_text, text=True, check=False)
            subprocess.run(["wl-copy", "--primary"], input=final_text, text=True, check=False)
            print("Copied to clipboard and primary selection.")
        else:
            print("wl-copy not found — clipboard not updated.")


if __name__ == "__main__":
    main()
