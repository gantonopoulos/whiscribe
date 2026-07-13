#!/usr/bin/env python3
"""whiscribe-tray — KDE/Wayland system tray front end for whiscribe.

Records and transcribes from the tray menu or via a global shortcut. The
shortcut is wired by binding a KDE custom shortcut to `whiscribe-tray --toggle`,
which talks to the running instance over a local socket (see README). Shared
recording/transcription logic lives in backend.py."""

import json
import os
import pathlib
import sys
import tempfile

# Ensure sibling modules import even when invoked through a symlink.
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

from PySide6.QtCore import Qt, QThread, Signal, QTimer, QPoint
from PySide6.QtGui import QIcon, QAction
from PySide6.QtWidgets import (
    QApplication, QSystemTrayIcon, QMenu, QFileDialog, QWidget, QLabel, QHBoxLayout,
)
from PySide6.QtNetwork import QLocalServer, QLocalSocket

import backend

SOCKET_NAME = "whiscribe-tray"
STATE_PATH = backend.CONFIG_DIR / "tray_state.json"

# Theme icon candidates per state (first that exists in the icon theme wins).
ICONS = {
    "idle": ["audio-input-microphone", "microphone-sensitivity-high"],
    "recording": ["media-record", "media-playback-recording", "audio-input-microphone"],
    "busy": ["chronometer", "view-refresh", "appointment-soon"],
}


def _icon(state: str) -> QIcon:
    for name in ICONS[state]:
        ic = QIcon.fromTheme(name)
        if not ic.isNull():
            return ic
    return QIcon.fromTheme("audio-input-microphone")


# ---------------------------------------------------------------------------
# Worker threads
# ---------------------------------------------------------------------------

class RecordWorker(QThread):
    """Prepares the device (switching BT profile if needed), records until
    stop() is called, restores the BT profile, and emits the WAV path."""

    done = Signal(str)      # wav_path
    failed = Signal(str)

    def __init__(self, device: backend.InputDevice):
        super().__init__()
        self.device = device
        self._recorder: backend.Recorder | None = None

    def stop(self) -> None:
        if self._recorder:
            self._recorder.stop()

    def run(self) -> None:
        try:
            source_name, bt_card, bt_saved = backend.prepare_device(self.device)
        except Exception as exc:  # noqa: BLE001 — surface any prep failure to the UI
            self.failed.emit(str(exc))
            return

        fd, tmp = tempfile.mkstemp(suffix=".wav", prefix="whiscribe_")
        os.close(fd)
        wav_path = pathlib.Path(tmp)

        try:
            self._recorder = backend.Recorder(source_name, wav_path)
            self._recorder.start()
            self._recorder.wait()
        except Exception as exc:  # noqa: BLE001
            wav_path.unlink(missing_ok=True)
            self.failed.emit(str(exc))
            return
        finally:
            if bt_card and bt_saved:
                try:
                    backend.switch_bt_profile(bt_card, bt_saved)
                except Exception:  # noqa: BLE001 — best-effort restore
                    pass

        if not wav_path.exists() or wav_path.stat().st_size < 1024:
            wav_path.unlink(missing_ok=True)
            self.failed.emit("No usable recording — was anything captured?")
            return

        self.done.emit(str(wav_path))


class TranscribeWorker(QThread):
    """Transcribes a WAV (GPU then CPU), post-processes, and emits the text.
    Always deletes the WAV."""

    done = Signal(str)      # transcript
    failed = Signal(str)

    def __init__(self, wav_path: str, cfg: dict,
                 model_path: pathlib.Path, vad_model: pathlib.Path | None):
        super().__init__()
        self.wav_path = wav_path
        self.cfg = cfg
        self.model_path = model_path
        self.vad_model = vad_model

    def run(self) -> None:
        wav = pathlib.Path(self.wav_path)
        try:
            raw = backend.transcribe_with_gpu_fallback(
                wav, self.model_path, self.cfg["threads"],
                self.cfg["language"] or None, vad_model=self.vad_model,
            )
        except Exception as exc:  # noqa: BLE001
            self.failed.emit(str(exc))
            return
        finally:
            wav.unlink(missing_ok=True)

        text = raw if self.cfg["timestamps"] else backend.strip_timestamps(raw)
        text = backend.collapse_repeats(text)
        if not text.strip():
            self.failed.emit("No speech detected.")
            return
        self.done.emit(text)


# ---------------------------------------------------------------------------
# Status overlay
# ---------------------------------------------------------------------------

class StatusWidget(QWidget):
    """Small borderless overlay near the tray showing state + elapsed time."""

    def __init__(self):
        super().__init__()
        self.setWindowFlags(
            Qt.WindowType.FramelessWindowHint
            | Qt.WindowType.WindowStaysOnTopHint
            | Qt.WindowType.Tool
        )
        self.setStyleSheet(
            "background-color: #1e1e2e; color: #cdd6f4; border-radius: 6px;"
        )
        layout = QHBoxLayout(self)
        layout.setContentsMargins(12, 7, 12, 7)

        self.indicator = QLabel("⬤")  # ⬤
        layout.addWidget(self.indicator)
        self.label = QLabel("Recording...")
        layout.addWidget(self.label)
        self.timer_label = QLabel("0:00")
        layout.addWidget(self.timer_label)

        self._elapsed = 0
        self._timer = QTimer(self)
        self._timer.timeout.connect(self._tick)

    def show_recording(self):
        self._elapsed = 0
        self.indicator.setStyleSheet("color: #f38ba8;")
        self.label.setText("Recording...")
        self.timer_label.setText("0:00")
        self.timer_label.show()
        self._timer.start(1000)
        self._show()

    def show_transcribing(self):
        self._timer.stop()
        self.indicator.setStyleSheet("color: #f9e2af;")
        self.label.setText("Transcribing...")
        self.timer_label.hide()
        self._show()

    def show_done(self, message: str):
        self._timer.stop()
        self.indicator.setStyleSheet("color: #a6e3a1;")
        self.label.setText(message)
        self.timer_label.hide()
        self._show()
        QTimer.singleShot(2500, self.hide)

    def show_error(self, message: str):
        self._timer.stop()
        self.indicator.setStyleSheet("color: #f38ba8;")
        self.label.setText(message)
        self.timer_label.hide()
        self._show()
        QTimer.singleShot(5000, self.hide)

    def _tick(self):
        self._elapsed += 1
        self.timer_label.setText(f"{self._elapsed // 60}:{self._elapsed % 60:02d}")

    def _show(self):
        self.adjustSize()
        screen = QApplication.primaryScreen()
        if screen:
            geom = screen.availableGeometry()
            self.move(QPoint(geom.right() - self.width() - 12,
                             geom.bottom() - self.height() - 12))
        self.show()


# ---------------------------------------------------------------------------
# Tray application
# ---------------------------------------------------------------------------

class WhisperTray(QWidget):
    """Owns the tray icon, menu, workers, and the toggle IPC server."""

    def __init__(self, app: QApplication):
        super().__init__()
        self.app = app
        self.cfg = backend.load_config()
        self.state = self._load_state()

        self._recording = False
        self._transcribing = False
        self._record_worker: RecordWorker | None = None
        self._transcribe_worker: TranscribeWorker | None = None
        self._output_file: str | None = None
        self._devices: list[backend.InputDevice] = []
        self._selected_device: backend.InputDevice | None = None

        self.status = StatusWidget()
        self._setup_tray()
        self._setup_ipc()

    # --- persistent tray state (device selection) -------------------------

    def _load_state(self) -> dict:
        try:
            return json.loads(STATE_PATH.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return {}

    def _save_state(self) -> None:
        try:
            STATE_PATH.parent.mkdir(parents=True, exist_ok=True)
            STATE_PATH.write_text(json.dumps(self.state, indent=2), encoding="utf-8")
        except OSError:
            pass

    # --- tray + menu ------------------------------------------------------

    def _setup_tray(self):
        self.tray = QSystemTrayIcon(self)
        self.tray.setIcon(_icon("idle"))
        self.tray.setToolTip("whiscribe — ready")
        self.tray.activated.connect(self._on_activated)

        self.menu = QMenu()

        self.act_clip = QAction("Record to clipboard", self)
        self.act_clip.triggered.connect(lambda: self._start(to_file=False))
        self.menu.addAction(self.act_clip)

        self.act_file = QAction("Record to file…", self)
        self.act_file.triggered.connect(lambda: self._start(to_file=True))
        self.menu.addAction(self.act_file)

        self.menu.addSeparator()

        self.act_stop = QAction("Stop recording", self)
        self.act_stop.setEnabled(False)
        self.act_stop.triggered.connect(self._stop)
        self.menu.addAction(self.act_stop)

        self.menu.addSeparator()

        self.device_menu = QMenu("Microphone", self.menu)
        self.device_menu.aboutToShow.connect(self._populate_devices)
        self.menu.addMenu(self.device_menu)
        self._populate_devices()  # also resolves the initial selection

        self.act_model = QAction("", self)
        self.act_model.setEnabled(False)
        self.menu.addAction(self.act_model)

        self.act_edit_cfg = QAction("Edit config…", self)
        self.act_edit_cfg.triggered.connect(self._edit_config)
        self.menu.addAction(self.act_edit_cfg)

        self.menu.addSeparator()

        act_quit = QAction("Quit", self)
        act_quit.triggered.connect(self._quit)
        self.menu.addAction(act_quit)

        self.menu.aboutToShow.connect(self._refresh_labels)
        self.tray.setContextMenu(self.menu)
        self._refresh_labels()
        self.tray.show()

    def _refresh_labels(self):
        self.cfg = backend.load_config()
        self.act_model.setText(f"Model: {self.cfg['model']}")

    def _populate_devices(self):
        try:
            self._devices = backend.list_input_devices()
        except RuntimeError as exc:
            self.status.show_error(str(exc))
            self._devices = []

        self.device_menu.clear()
        saved = self.state.get("device_label")

        # Re-resolve the current selection against the fresh list.
        self._selected_device = None
        for dev in self._devices:
            if dev.label == saved:
                self._selected_device = dev
        if self._selected_device is None and self._devices:
            self._selected_device = self._devices[0]

        if not self._devices:
            empty = QAction("(no input devices)", self.device_menu)
            empty.setEnabled(False)
            self.device_menu.addAction(empty)
            return

        for dev in self._devices:
            act = QAction(dev.label, self.device_menu)
            act.setCheckable(True)
            act.setChecked(dev is self._selected_device)
            act.triggered.connect(lambda _checked=False, d=dev: self._select_device(d))
            self.device_menu.addAction(act)

    def _select_device(self, device: backend.InputDevice):
        self._selected_device = device
        self.state["device_label"] = device.label
        self._save_state()

    def _edit_config(self):
        backend.load_config()  # ensure the file exists
        import subprocess
        subprocess.Popen(["xdg-open", str(backend.CONFIG_PATH)])

    # --- activation / toggle ---------------------------------------------

    def _on_activated(self, reason):
        if reason == QSystemTrayIcon.ActivationReason.Trigger:
            self.toggle()

    def toggle(self):
        """Global-shortcut / left-click action: start (to clipboard) or stop."""
        if self._recording:
            self._stop()
        elif not self._transcribing:
            self._start(to_file=False)

    # --- recording --------------------------------------------------------

    def _start(self, to_file: bool):
        if self._recording or self._transcribing:
            return
        if self._selected_device is None:
            self.status.show_error("No microphone selected")
            return

        self.cfg = backend.load_config()
        model_path = backend.resolve_model_path(self.cfg["model"])
        if not model_path.exists():
            self.status.show_error(f"Model not found: {self.cfg['model']}")
            return

        self._output_file = None
        if to_file:
            default_dir = pathlib.Path.home()
            path, _ = QFileDialog.getSaveFileName(
                None, "Save transcription", str(default_dir / "transcript.txt"),
                "Text files (*.txt)",
            )
            if not path:
                return
            self._output_file = path

        self._recording = True
        self._set_busy_ui("recording")
        self.status.show_recording()

        self._record_worker = RecordWorker(self._selected_device)
        self._record_worker.done.connect(self._on_recorded)
        self._record_worker.failed.connect(self._on_error)
        self._record_worker.start()

    def _stop(self):
        if self._recording and self._record_worker:
            self._record_worker.stop()

    def _on_recorded(self, wav_path: str):
        self._recording = False
        self._transcribing = True
        self._set_busy_ui("busy")
        self.status.show_transcribing()

        model_path = backend.resolve_model_path(self.cfg["model"])
        vad_model = backend.DEFAULT_VAD_MODEL if self.cfg["vad"] else None

        self._transcribe_worker = TranscribeWorker(wav_path, self.cfg, model_path, vad_model)
        self._transcribe_worker.done.connect(self._on_transcribed)
        self._transcribe_worker.failed.connect(self._on_error)
        self._transcribe_worker.start()

    def _on_transcribed(self, text: str):
        self._reset_idle()
        if self._output_file:
            try:
                p = pathlib.Path(self._output_file)
                p.parent.mkdir(parents=True, exist_ok=True)
                p.write_text(text + "\n", encoding="utf-8")
                self.status.show_done(f"Saved: {p.name}")
            except OSError as exc:
                self.status.show_error(f"Save failed: {exc}")
        else:
            if backend.set_clipboard(text):
                self.status.show_done("Copied to clipboard")
            else:
                self.status.show_error("Clipboard unavailable")

    def _on_error(self, message: str):
        self._reset_idle()
        self.status.show_error(message)

    # --- UI state helpers -------------------------------------------------

    def _set_busy_ui(self, state: str):
        self.tray.setIcon(_icon(state))
        self.tray.setToolTip(f"whiscribe — {state}")
        self.act_stop.setEnabled(state == "recording")
        self.act_clip.setEnabled(False)
        self.act_file.setEnabled(False)

    def _reset_idle(self):
        self._recording = False
        self._transcribing = False
        self.tray.setIcon(_icon("idle"))
        self.tray.setToolTip("whiscribe — ready")
        self.act_stop.setEnabled(False)
        self.act_clip.setEnabled(True)
        self.act_file.setEnabled(True)

    # --- IPC (single instance + global-shortcut toggle) -------------------

    def _setup_ipc(self):
        self.server = QLocalServer(self)
        QLocalServer.removeServer(SOCKET_NAME)  # clear a stale socket
        self.server.listen(SOCKET_NAME)
        self.server.newConnection.connect(self._on_ipc)

    def _on_ipc(self):
        conn = self.server.nextPendingConnection()
        if conn is None:
            return
        if conn.waitForReadyRead(500):
            cmd = bytes(conn.readAll().data()).decode(errors="ignore").strip()
            if cmd == "toggle":
                self.toggle()
        conn.disconnectFromServer()

    def _quit(self):
        if self._record_worker and self._record_worker.isRunning():
            self._record_worker.stop()
            self._record_worker.wait(3000)
        if self._transcribe_worker and self._transcribe_worker.isRunning():
            self._transcribe_worker.wait(5000)
        self.server.close()
        self.status.hide()
        self.tray.hide()
        self.app.quit()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def _forward_toggle(send: bool) -> bool:
    """Connect to a running instance (if any). When connected and `send` is set,
    tell it to toggle. Returns True if an instance was reached."""
    sock = QLocalSocket()
    sock.connectToServer(SOCKET_NAME)
    if not sock.waitForConnected(300):
        return False
    if send:
        sock.write(b"toggle")
        sock.waitForBytesWritten(400)
    sock.disconnectFromServer()
    return True


def main():
    app = QApplication(sys.argv)
    app.setQuitOnLastWindowClosed(False)
    app.setApplicationName("whiscribe-tray")

    toggle = "--toggle" in sys.argv[1:]

    # Single-instance: a running instance handles --toggle and rejects a second launch.
    if _forward_toggle(send=toggle):
        if not toggle:
            print("whiscribe-tray is already running.")
        return
    if toggle:
        print("whiscribe-tray is not running.")
        return

    if not QSystemTrayIcon.isSystemTrayAvailable():
        sys.exit("No system tray available.")

    tray = WhisperTray(app)  # noqa: F841 — kept alive by the event loop
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
