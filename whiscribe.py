#!/usr/bin/env python3
"""whiscribe — voice recording + transcription via whisper.cpp"""

import argparse
import curses
import datetime
import os
import pathlib
import re
import shutil
import subprocess
import sys
import time
from typing import NamedTuple


SCRIPT_DIR = pathlib.Path(__file__).parent.resolve()
DEFAULT_MODEL = SCRIPT_DIR / "llm_models" / "ggml-small.bin"


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
        ["arecord", "-D", "pulse", "-f", "cd", "-t", "wav", str(output_path)],
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
) -> str:
    """Run whisper-cli, stream each transcript segment to the terminal, return full text."""
    cmd = [
        "whisper-cli",
        "-m", str(model_path),
        "-f", str(wav_path),
        "-t", str(threads),
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
        sys.exit(f"whisper-cli exited with code {proc.returncode}")

    return "\n".join(lines)


def strip_timestamps(text: str) -> str:
    return re.sub(r"\[[\d:.]+ --> [\d:.]+\]\s*", "", text).strip()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def build_stem(output_dir: pathlib.Path, filename: str | None) -> pathlib.Path:
    if filename:
        return output_dir / filename
    stamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    return output_dir / f"whiscribe_{stamp}"


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
        "-o", "--output-dir", type=pathlib.Path, default=pathlib.Path.cwd(),
        metavar="DIR",
        help="Directory where WAV and TXT files are saved",
    )
    parser.add_argument(
        "-n", "--filename",
        metavar="NAME",
        help="Base filename for outputs (no extension); default: whiscribe_YYYYMMDD_HHMMSS",
    )
    parser.add_argument(
        "-t", "--threads", type=int, default=4,
        help="Number of threads for whisper inference",
    )
    parser.add_argument(
        "--gpu", dest="no_gpu", action="store_false",
        help="Enable GPU inference (default: CPU only)",
    )
    parser.set_defaults(no_gpu=True)
    parser.add_argument(
        "-l", "--language",
        metavar="LANG",
        help="Language code hint for whisper (e.g. 'en', 'el'); auto-detect if omitted",
    )
    parser.add_argument(
        "--plain-text", action="store_true",
        help="Strip timestamps from the saved text file (default: keep them)",
    )
    parser.add_argument(
        "--clip", choices=["path", "content"], default="path",
        help="What to copy to clipboard: 'path' = relative path to .txt file (default), "
             "'content' = full transcript text",
    )

    args = parser.parse_args()

    if not args.model.exists():
        sys.exit(f"Model not found: {args.model}")

    args.output_dir.mkdir(parents=True, exist_ok=True)

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

    # --- file paths ---
    stem = build_stem(args.output_dir, args.filename)
    wav_path = stem.with_suffix(".wav")
    txt_path = stem.with_suffix(".txt")

    # --- record; always restore BT profile in finally ---
    try:
        record(source_name, wav_path)
    finally:
        if bt_card and bt_saved_profile:
            print("Restoring Bluetooth profile...")
            switch_bt_profile(bt_card, bt_saved_profile)

    if not wav_path.exists() or wav_path.stat().st_size < 1024:
        sys.exit("No usable recording — WAV file is missing or too small.")

    # --- transcribe ---
    print("\nTranscribing...")
    raw_text = transcribe(wav_path, args.model, args.threads, args.no_gpu, args.language)
    print()

    if not raw_text:
        print("Warning: whisper returned empty output.")

    saved_text = strip_timestamps(raw_text) if args.plain_text else raw_text

    txt_path.write_text(saved_text + "\n", encoding="utf-8")
    print(f"Text : {txt_path}")
    print(f"Audio: {wav_path}")

    if shutil.which("wl-copy"):
        if args.clip == "path":
            try:
                clip_value = str(txt_path.relative_to(pathlib.Path.cwd()))
            except ValueError:
                clip_value = str(txt_path)
        else:
            clip_value = saved_text
        subprocess.run(["wl-copy"], input=clip_value, text=True, check=False)
        print(f"Copied to clipboard: {'file path' if args.clip == 'path' else 'transcript text'}")


if __name__ == "__main__":
    main()
