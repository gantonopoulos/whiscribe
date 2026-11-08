"""whiscribe backend — device discovery, recording, and transcription via
whisper.cpp. Pure logic with no UI: shared by the CLI (cli.py) and the system
tray app (tray.py). Failures raise exceptions rather than exiting so a GUI can
recover."""

import os
import pathlib
import re
import shutil
import signal
import subprocess
import threading
import time
import tomllib
from datetime import datetime, timedelta
from typing import NamedTuple


class Cancelled(Exception):
    """Raised by transcription when a caller-supplied cancel event fires."""


SCRIPT_DIR = pathlib.Path(__file__).parent.resolve()

# Whisper models live here; the config file names one by filename.
MODELS_DIR = SCRIPT_DIR / "llm_models"

# Every recording's WAV is kept here (finished or cancelled); the transcript
# text is never stored. Named by start time, e.g. 2026-11-08_14-03-22.wav.
RECORDINGS_DIR = SCRIPT_DIR / "recordings"
RECORDING_FMT = "%Y-%m-%d_%H-%M-%S"

# Voice Activity Detection model. Auto-enabled when present; download from
# https://huggingface.co/ggml-org/whisper-vad and place in llm_models/.
DEFAULT_VAD_MODEL = MODELS_DIR / "ggml-silero-v5.1.2.bin"

_WHISPER_HF = "https://huggingface.co/ggerganov/whisper.cpp/resolve/main/"
_VAD_HF = "https://huggingface.co/ggml-org/whisper-vad/resolve/main/"


class CatalogEntry(NamedTuple):
    url: str
    size: str          # human-readable, for display only
    description: str


# Downloadable models, in display order. Keys are filenames inside llm_models/.
MODEL_CATALOG: dict[str, CatalogEntry] = {
    "ggml-large-v3.bin": CatalogEntry(
        _WHISPER_HF + "ggml-large-v3.bin", "3.1 GB",
        "best accuracy (needs a capable GPU)"),
    "ggml-large-v3-turbo.bin": CatalogEntry(
        _WHISPER_HF + "ggml-large-v3-turbo.bin", "1.6 GB",
        "near-large accuracy, much faster"),
    "ggml-small.bin": CatalogEntry(
        _WHISPER_HF + "ggml-small.bin", "466 MB",
        "lightweight CPU-friendly fallback"),
    "ggml-medium.bin": CatalogEntry(
        _WHISPER_HF + "ggml-medium.bin", "1.5 GB", "between small and large"),
    "ggml-base.bin": CatalogEntry(
        _WHISPER_HF + "ggml-base.bin", "142 MB", "fast, modest accuracy"),
    "ggml-tiny.bin": CatalogEntry(
        _WHISPER_HF + "ggml-tiny.bin", "75 MB", "fastest, lowest accuracy"),
}

VAD_MODEL_NAME = DEFAULT_VAD_MODEL.name
VAD_CATALOG_ENTRY = CatalogEntry(
    _VAD_HF + VAD_MODEL_NAME, "~1 MB",
    "Silero voice activity detection; curbs hallucinations on silence")

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
    "delete_on_start": True,       # wipe all saved recordings at app start
    "auto_delete": True,           # periodically delete old recordings
    "auto_delete_hours": 24,       # age threshold (and check is hourly)
    "warn_size_mb": 1024,          # warn when saved recordings exceed this
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

# Every recording's audio (WAV) is kept in the recordings/ directory next to
# whiscribe.py, whether it finished or was cancelled. Delete all of them each
# time the tray app starts.
delete_on_start = true

# While running, delete recordings older than auto_delete_hours (checked
# hourly and before each recording).
auto_delete = true
auto_delete_hours = 24

# Warn when saved recordings use more than this many MB (1024 = 1 GB).
warn_size_mb = 1024
"""

# Per-key help for GUIs (tooltips); mirrors the comments in CONFIG_TEMPLATE.
CONFIG_HELP = {
    "model": "Whisper model filename, looked up in the llm_models/ directory.",
    "threads": "Threads for whisper inference.",
    "language": 'Language hint, e.g. "en" or "el". Leave empty for auto-detect.',
    "vad": "Strip non-speech regions before transcription (needs the Silero "
           "VAD model). Strongly recommended — reduces hallucinated output.",
    "timestamps": "Include whisper timestamps in the output.",
    "delete_on_start": "Delete all saved recordings each time the tray app starts.",
    "auto_delete": "While running, periodically delete recordings older than the "
                   "age limit.",
    "auto_delete_hours": "Recordings older than this many hours are deleted "
                         "(checked hourly and before each recording).",
    "warn_size_mb": "Show a warning when saved recordings exceed this size in MB "
                    "(1024 = 1 GB).",
}

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
        raise RuntimeError("pactl not found — is PulseAudio/PipeWire running?")

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


def prepare_device(device: InputDevice) -> tuple[str, str | None, str | None]:
    """
    Ensure the device is ready for recording.
    For BT cards in A2DP mode: switch profile and wait for the input source to appear.
    Returns (source_name, bt_card_to_restore, profile_to_restore).
    Raises RuntimeError if a required Bluetooth switch times out.
    """
    if device.source_name:
        return device.source_name, None, None

    switch_bt_profile(device.bt_card_name, device.bt_target_profile)
    source_name = find_bt_source_after_switch(device.bt_card_name)
    if not source_name:
        raise RuntimeError(
            "Timed out waiting for headset audio source after Bluetooth profile switch."
        )
    return source_name, device.bt_card_name, device.bt_saved_profile


# ---------------------------------------------------------------------------
# Recording
# ---------------------------------------------------------------------------

class Recorder:
    """arecord-backed recorder. start() begins capture, wait() blocks until the
    process ends (used by the CLI's Ctrl-C flow), and stop() ends capture
    cleanly via SIGINT (used by the GUI, whose arecord is not in the terminal's
    process group)."""

    def __init__(self, source_name: str, output_path: pathlib.Path):
        self.source_name = source_name
        self.output_path = pathlib.Path(output_path)
        self._proc: subprocess.Popen | None = None

    def start(self) -> None:
        env = {**os.environ, "PULSE_SOURCE": self.source_name}
        self._proc = subprocess.Popen(
            ["arecord", "-D", "pulse",
             "-f", "S16_LE", "-r", RECORD_RATE, "-c", RECORD_CHANNELS,
             "-t", "wav", str(self.output_path)],
            env=env,
            stderr=subprocess.DEVNULL,
        )

    def wait(self) -> None:
        if self._proc:
            self._proc.wait()

    def stop(self) -> None:
        if self._proc and self._proc.poll() is None:
            # SIGINT lets arecord flush the WAV header before exiting.
            self._proc.send_signal(signal.SIGINT)
            self._proc.wait()


def record(source_name: str, output_path: pathlib.Path) -> None:
    """Blocking record for the CLI: Ctrl-C stops cleanly."""
    print("\nRecording... Press Ctrl+C to stop.\n")
    rec = Recorder(source_name, output_path)
    rec.start()
    try:
        rec.wait()
    except KeyboardInterrupt:
        rec.stop()


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
    cancel_event: "threading.Event | None" = None,
) -> str | None:
    """Run whisper-cli, stream each transcript segment to stdout, return full text
    or None on failure. Raises Cancelled if cancel_event fires mid-run."""
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

    # A watcher terminates whisper-cli promptly on cancel — even during the
    # (output-free) model-load phase, which a per-line check would miss.
    watcher: threading.Thread | None = None
    if cancel_event is not None:
        def _watch():
            while proc.poll() is None:
                if cancel_event.wait(0.2):
                    proc.terminate()
                    return
        watcher = threading.Thread(target=_watch, daemon=True)
        watcher.start()

    lines: list[str] = []
    for raw_line in proc.stdout:
        line = raw_line.rstrip("\n")
        if line:
            print(line)
            lines.append(line)
    proc.wait()
    if watcher is not None:
        watcher.join(timeout=1)

    if cancel_event is not None and cancel_event.is_set():
        raise Cancelled()
    if proc.returncode != 0:
        return None

    return "\n".join(lines)


def transcribe_with_gpu_fallback(
    wav_path: pathlib.Path,
    model_path: pathlib.Path,
    threads: int,
    language: str | None,
    vad_model: pathlib.Path | None = None,
    cancel_event: "threading.Event | None" = None,
) -> str:
    """Try GPU transcription first; fall back to CPU if it fails.
    Raises RuntimeError if both fail, or Cancelled if aborted (no CPU retry)."""
    print("\nTranscribing (GPU)...")
    result = transcribe(wav_path, model_path, threads, no_gpu=False,
                        language=language, vad_model=vad_model, cancel_event=cancel_event)
    if result is not None:
        return result

    print("GPU transcription failed, retrying on CPU...")
    result = transcribe(wav_path, model_path, threads, no_gpu=True,
                        language=language, vad_model=vad_model, cancel_event=cancel_event)
    if result is None:
        raise RuntimeError("Transcription failed on both GPU and CPU.")
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
# Model downloads and config writing
# ---------------------------------------------------------------------------

def catalog_entry(name: str) -> CatalogEntry:
    """Catalog entry for a model filename; 'vad' aliases the VAD model."""
    if name == "vad" or name == VAD_MODEL_NAME:
        return VAD_CATALOG_ENTRY
    try:
        return MODEL_CATALOG[name]
    except KeyError:
        raise ValueError(f"Unknown model: {name}") from None


def model_present(name: str) -> bool:
    return resolve_model_path(VAD_MODEL_NAME if name == "vad" else name).exists()


def download_model(
    name: str,
    progress_cb=None,
    cancel_event: threading.Event | None = None,
) -> pathlib.Path:
    """Download a catalog model into llm_models/. progress_cb(done, total) is
    called as bytes arrive (total is 0 when unknown). Streams to <name>.part and
    renames on success; the partial file is removed on failure or cancel.
    Raises Cancelled if cancel_event fires, OSError on network/disk errors."""
    import urllib.request

    entry = catalog_entry(name)
    dest = resolve_model_path(VAD_MODEL_NAME if name == "vad" else name)
    part = dest.with_name(dest.name + ".part")
    dest.parent.mkdir(parents=True, exist_ok=True)

    try:
        req = urllib.request.Request(entry.url, headers={"User-Agent": "whiscribe"})
        with urllib.request.urlopen(req, timeout=30) as resp, open(part, "wb") as out:
            total = int(resp.headers.get("Content-Length") or 0)
            done = 0
            while True:
                if cancel_event is not None and cancel_event.is_set():
                    raise Cancelled()
                chunk = resp.read(1 << 20)
                if not chunk:
                    break
                out.write(chunk)
                done += len(chunk)
                if progress_cb:
                    progress_cb(done, total)
        if total and done != total:
            raise OSError(f"Incomplete download: got {done} of {total} bytes")
        part.replace(dest)
    except BaseException:
        part.unlink(missing_ok=True)
        raise
    return dest


def _toml_value(value) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, int):
        return str(value)
    escaped = str(value).replace("\\", "\\\\").replace('"', '\\"')
    return f'"{escaped}"'


def save_config(updates: dict) -> None:
    """Write key = value changes into config.toml, keeping comments and order.
    Keys not already present are appended. The result is re-parsed before it
    replaces the file, so a bad write can't corrupt the config."""
    if not CONFIG_PATH.exists():
        load_config()  # creates the commented template
    lines = CONFIG_PATH.read_text(encoding="utf-8").splitlines()

    for key, value in updates.items():
        new_line = f"{key} = {_toml_value(value)}"
        pattern = re.compile(rf"^\s*{re.escape(key)}\s*=")
        for i, line in enumerate(lines):
            if pattern.match(line):
                lines[i] = new_line
                break
        else:
            lines.append(new_line)

    text = "\n".join(lines) + "\n"
    tomllib.loads(text)  # raises TOMLDecodeError before anything is written
    tmp = CONFIG_PATH.with_name(CONFIG_PATH.name + ".tmp")
    tmp.write_text(text, encoding="utf-8")
    tmp.replace(CONFIG_PATH)


# ---------------------------------------------------------------------------
# Saved recordings
# ---------------------------------------------------------------------------

def new_recording_path() -> pathlib.Path:
    """Path for a new recording, named by the current time (second granularity)."""
    RECORDINGS_DIR.mkdir(parents=True, exist_ok=True)
    return RECORDINGS_DIR / (datetime.now().strftime(RECORDING_FMT) + ".wav")


def list_recordings() -> list[tuple[datetime, pathlib.Path]]:
    """Saved recordings, newest first. The time comes from the filename (not
    mtime, which changes when files are copied or touched)."""
    if not RECORDINGS_DIR.is_dir():
        return []
    found = []
    for p in RECORDINGS_DIR.glob("*.wav"):
        try:
            found.append((datetime.strptime(p.stem, RECORDING_FMT), p))
        except ValueError:
            continue   # not one of ours; leave it alone
    return sorted(found, reverse=True)


def recordings_size() -> int:
    """Total bytes used by saved recordings."""
    total = 0
    for _, p in list_recordings():
        try:
            total += p.stat().st_size
        except OSError:
            pass
    return total


def delete_recording(path: pathlib.Path) -> None:
    path.unlink(missing_ok=True)


def purge_recordings(
    max_age_hours: float | None = None,
    keep: tuple | list = (),
) -> int:
    """Delete saved recordings older than max_age_hours (None = all of them).
    Paths in `keep` (in-flight recordings) are never touched. Returns the
    number deleted."""
    cutoff = (None if max_age_hours is None
              else datetime.now() - timedelta(hours=max_age_hours))
    keep_set = {pathlib.Path(k).resolve() for k in keep}
    deleted = 0
    for when, path in list_recordings():
        if path.resolve() in keep_set:
            continue
        if cutoff is not None and when >= cutoff:
            continue
        try:
            path.unlink()
            deleted += 1
        except OSError:
            pass
    return deleted


def size_warning(cfg: dict) -> str | None:
    """Message if saved recordings exceed the configured limit, else None."""
    used = recordings_size()
    limit = int(cfg["warn_size_mb"]) * 1024 * 1024
    if limit > 0 and used > limit:
        return (f"Saved recordings use {used / 1024**3:.1f} GB "
                f"(limit {limit / 1024**3:.1f} GB) — clear them in "
                f"Previous transcriptions")
    return None
