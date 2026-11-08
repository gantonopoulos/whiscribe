"""whiscribe CLI — interactive device picker, record, transcribe. The heavy
lifting lives in backend.py; this module is the terminal front end."""

import argparse
import curses
import pathlib
import sys
import threading

import backend
from backend import (
    CONFIG_PATH,
    DEFAULT_VAD_MODEL,
    MODELS_DIR,
    InputDevice,
)


# ---------------------------------------------------------------------------
# Interactive curses picker
# ---------------------------------------------------------------------------

def _run_picker(stdscr: "curses.window", items: list[str]) -> int:
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
# Model management
# ---------------------------------------------------------------------------

def list_models(current: str) -> None:
    rows = [(n, e) for n, e in backend.MODEL_CATALOG.items()]
    rows.append(("vad", backend.VAD_CATALOG_ENTRY))
    for name, entry in rows:
        status = "present" if backend.model_present(name) else "missing"
        mark = "*" if name == current else " "
        print(f"{mark} {name:<26} {status:<8} {entry.size:>7}  {entry.description}")
    print(f"\n* = configured model. Models directory: {MODELS_DIR}")


def download_model(name: str) -> None:
    try:
        entry = backend.catalog_entry(name)
    except ValueError as exc:
        sys.exit(f"{exc}\nRun --list-models to see what is available.")
    if backend.model_present(name):
        print(f"Already present: {name}")
        return

    print(f"Downloading {name} ({entry.size}) …")

    def progress(done: int, total: int) -> None:
        if total:
            pct = done * 100 // total
            print(f"\r  {pct:3d}%  {done >> 20} / {total >> 20} MB", end="", flush=True)
        else:
            print(f"\r  {done >> 20} MB", end="", flush=True)

    cancel = threading.Event()
    try:
        path = backend.download_model(name, progress, cancel)
    except KeyboardInterrupt:
        sys.exit("\nDownload cancelled.")
    except OSError as exc:
        sys.exit(f"\nDownload failed: {exc}")
    print(f"\nSaved: {path}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    cfg = backend.load_config()
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

    parser.add_argument(
        "--list-models", action="store_true",
        help="Show downloadable models and whether each is present, then exit",
    )
    parser.add_argument(
        "--download-model", metavar="NAME",
        help="Download a model into llm_models/ (use 'vad' for the VAD model), then exit",
    )

    parser.add_argument(
        "--cpu", action="store_true",
        help="Force CPU-only transcription for this run (config: gpu = \"off\")",
    )
    parser.add_argument(
        "--gpu-device", type=int, default=cfg["gpu_device"], metavar="N",
        help="Which GPU to use when there are several (0 = first)",
    )

    args = parser.parse_args()

    if args.list_models:
        list_models(cfg["model"])
        return
    if args.download_model:
        download_model(args.download_model)
        return

    # --- resolve the model: CLI overrides config; must exist in llm_models/ ---
    model_path = backend.resolve_model_path(args.model or cfg["model"])
    if not model_path.exists():
        have = backend.available_models()
        listing = "\n  ".join(have) if have else "(none)"
        sys.exit(
            f"Model not found: {model_path}\n"
            f"Set 'model' in {CONFIG_PATH} to one of the models in {MODELS_DIR}:\n"
            f"  {listing}\n"
            f"Or fetch one with: whiscribe --download-model {model_path.name}  "
            f"(see --list-models)"
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
    try:
        devices = backend.list_input_devices()
    except RuntimeError as exc:
        sys.exit(str(exc))
    if not devices:
        sys.exit("No audio input devices found.")

    device = pick_device(devices)
    if device is None:
        print("Aborted.")
        sys.exit(0)

    # --- prepare device (switches BT profile if needed) ---
    try:
        source_name, bt_card, bt_saved_profile = backend.prepare_device(device)
    except RuntimeError as exc:
        sys.exit(str(exc))

    # --- the WAV is kept in recordings/ (see auto_delete / warn_size_mb) ---
    if cfg["auto_delete"]:
        backend.purge_recordings(max_age_hours=cfg["auto_delete_hours"])
    warning = backend.size_warning(cfg)
    if warning:
        print(f"Warning: {warning}")
    wav_path = backend.new_recording_path()

    # --- record; always restore BT profile in finally ---
    try:
        backend.record(source_name, wav_path)
    finally:
        if bt_card and bt_saved_profile:
            print("Restoring Bluetooth profile...")
            backend.switch_bt_profile(bt_card, bt_saved_profile)

    if not wav_path.exists() or wav_path.stat().st_size < 1024:
        wav_path.unlink(missing_ok=True)
        sys.exit("No usable recording — WAV file is missing or too small.")
    print(f"Recording kept: {wav_path}")

    # --- transcribe (GPU first unless disabled, CPU fallback) ---
    use_gpu = cfg["gpu"] != "off" and not args.cpu
    gpus = backend.detect_gpus() if use_gpu else []
    print(f"Compute: {backend.compute_summary({**cfg, 'gpu': 'auto' if use_gpu else 'off', 'gpu_device': args.gpu_device}, gpus)}")
    try:
        result = backend.transcribe_with_gpu_fallback(
            wav_path, model_path, args.threads, args.language, vad_model=vad_model,
            use_gpu=use_gpu, gpu_device=args.gpu_device,
            expect_gpu=use_gpu and gpus != [])
    except RuntimeError as exc:
        sys.exit(str(exc))
    raw_text = result.text
    if result.fell_back:
        print("Note: GPU failed — transcribed on CPU instead.")
    print()

    if not raw_text:
        print("Warning: whisper returned empty output.")

    final_text = raw_text if args.timestamps else backend.strip_timestamps(raw_text)
    final_text = backend.collapse_repeats(final_text)

    # --- save to file if -o given ---
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(final_text + "\n", encoding="utf-8")
        print(f"Saved: {args.output}")

    # --- clipboard: always in default mode; only with --clip when -o is given ---
    copy_to_clipboard = (args.output is None) or args.clip
    if copy_to_clipboard:
        if backend.set_clipboard(final_text):
            print("Copied to clipboard.")
        else:
            print("Clipboard not updated — install wl-clipboard (or start KDE Klipper).")


if __name__ == "__main__":
    main()
