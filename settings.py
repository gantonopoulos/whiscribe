"""whiscribe settings window — pick/download the Whisper model and edit the
other config keys. Reads and writes the shared config.toml through backend, so
the CLI and tray see the same state."""

import threading

from PySide6.QtCore import QThread, Signal, Qt
from PySide6.QtGui import QPalette
from PySide6.QtWidgets import (
    QButtonGroup, QCheckBox, QComboBox, QDialog, QDialogButtonBox, QFormLayout, QGroupBox,
    QHBoxLayout, QLabel, QLineEdit, QMessageBox, QProgressBar, QPushButton,
    QRadioButton, QSpinBox, QVBoxLayout,
)

import backend


class DownloadWorker(QThread):
    """Downloads one catalog model. Signals are named done/failed (not
    finished, which would shadow QThread.finished)."""

    progress = Signal(int, int)   # bytes done, total (0 = unknown)
    done = Signal(str)            # model name
    failed = Signal(str)

    def __init__(self, name: str):
        super().__init__()
        self.name = name
        self._cancel = threading.Event()

    def request_cancel(self) -> None:
        self._cancel.set()

    def run(self) -> None:
        try:
            backend.download_model(self.name, self.progress.emit, self._cancel)
        except backend.Cancelled:
            return
        except Exception as exc:  # noqa: BLE001
            self.failed.emit(str(exc))
            return
        self.done.emit(self.name)


class SettingsDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("whiscribe settings")
        self.setMinimumWidth(460)
        self.cfg = backend.load_config()
        self._worker: DownloadWorker | None = None
        self._radios: dict[str, QRadioButton] = {}
        self._current = self.cfg["model"]

        root = QVBoxLayout(self)
        root.addWidget(self._build_models())
        root.addWidget(self._build_options())
        root.addWidget(self._build_compute())
        root.addWidget(self._build_recordings())

        # Shared progress row for whichever download is running.
        self.progress_label = QLabel()
        self.progress_bar = QProgressBar()
        self.cancel_btn = QPushButton("Cancel download")
        self.cancel_btn.clicked.connect(self._cancel_download)
        row = QHBoxLayout()
        row.addWidget(self.progress_bar, 1)
        row.addWidget(self.cancel_btn)
        self.progress_box = QVBoxLayout()
        self.progress_box.addWidget(self.progress_label)
        self.progress_box.addLayout(row)
        root.addLayout(self.progress_box)
        self._show_progress(False)

        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Save
            | QDialogButtonBox.StandardButton.Cancel
        )
        buttons.accepted.connect(self._save)
        buttons.rejected.connect(self.reject)
        root.addWidget(buttons)

    # --- building ---------------------------------------------------------

    def _build_models(self) -> QGroupBox:
        box = QGroupBox("Transcription model")
        layout = QVBoxLayout(box)
        layout.addWidget(self._hint(backend.CONFIG_HELP["model"]))

        self.group = QButtonGroup(self)
        names = list(backend.MODEL_CATALOG)
        if self._current not in names:   # custom model set by hand in config
            names.append(self._current)
        for name in names:
            radio = QRadioButton()
            radio.setProperty("model", name)
            if name in backend.MODEL_CATALOG:
                entry = backend.MODEL_CATALOG[name]
                radio.setToolTip(f"{entry.description} ({entry.size})")
            self.group.addButton(radio)
            self._radios[name] = radio
            layout.addWidget(radio)
            radio.setChecked(name == self._current)
            self._refresh_radio(name)
        self.group.buttonClicked.connect(self._on_model_clicked)
        return box

    def _build_options(self) -> QGroupBox:
        box = QGroupBox("Options")
        form = QFormLayout(box)

        self.threads = QSpinBox()
        self.threads.setRange(1, 64)
        self.threads.setValue(int(self.cfg["threads"]))
        self.threads.setToolTip(backend.CONFIG_HELP["threads"])
        form.addRow("Threads", self.threads)

        self.language = QLineEdit(str(self.cfg["language"]))
        self.language.setPlaceholderText("auto-detect")
        self.language.setToolTip(backend.CONFIG_HELP["language"])
        form.addRow("Language", self.language)

        self.vad = QCheckBox("Filter non-speech (VAD)")
        self.vad.setChecked(bool(self.cfg["vad"]))
        self.vad.setToolTip(backend.CONFIG_HELP["vad"])
        self.vad_status = QLabel()
        self.vad_btn = QPushButton("Download VAD model")
        self.vad_btn.clicked.connect(lambda: self._start_download("vad"))
        vad_row = QHBoxLayout()
        vad_row.addWidget(self.vad)
        vad_row.addWidget(self.vad_status)
        vad_row.addStretch(1)
        vad_row.addWidget(self.vad_btn)
        form.addRow(vad_row)
        self._refresh_vad()

        self.timestamps = QCheckBox("Include timestamps")
        self.timestamps.setChecked(bool(self.cfg["timestamps"]))
        self.timestamps.setToolTip(backend.CONFIG_HELP["timestamps"])
        form.addRow(self.timestamps)
        return box

    def _build_compute(self) -> QGroupBox:
        box = QGroupBox("Compute")
        form = QFormLayout(box)

        self.gpu_mode = QComboBox()
        self.gpu_mode.addItem("Auto (use GPU if available)", "auto")
        self.gpu_mode.addItem("CPU only", "off")
        self.gpu_mode.setCurrentIndex(0 if self.cfg["gpu"] != "off" else 1)
        self.gpu_mode.setToolTip(backend.CONFIG_HELP["gpu"])
        form.addRow("Mode", self.gpu_mode)

        self.gpu_device = QComboBox()
        self.gpu_device.setToolTip(backend.CONFIG_HELP["gpu_device"])
        self.gpu_status = self._hint("")
        redetect = QPushButton("Re-detect")
        redetect.clicked.connect(self._detect_gpus)
        row = QHBoxLayout()
        row.addWidget(self.gpu_device, 1)
        row.addWidget(redetect)
        form.addRow("GPU", row)
        form.addRow(self.gpu_status)

        self.gpu_mode.currentIndexChanged.connect(self._update_gpu_enabled)
        self._detect_gpus()
        return box

    def _detect_gpus(self) -> None:
        gpus = backend.detect_gpus()
        self.gpu_device.clear()
        for g in gpus or []:
            self.gpu_device.addItem(g.name, g.index)
        i = self.gpu_device.findData(int(self.cfg["gpu_device"]))
        self.gpu_device.setCurrentIndex(max(i, 0))
        if gpus is None:
            self.gpu_status.setText("vulkaninfo not found — install vulkan-tools "
                                    "to detect GPUs. Auto will still try the GPU.")
        elif not gpus:
            self.gpu_status.setText("No GPU found; transcription will use the CPU.")
        else:
            self.gpu_status.setText(f"{len(gpus)} GPU(s) detected.")
        self._update_gpu_enabled()

    def _update_gpu_enabled(self) -> None:
        self.gpu_device.setEnabled(
            self.gpu_mode.currentData() == "auto" and self.gpu_device.count() > 1)

    def _build_recordings(self) -> QGroupBox:
        box = QGroupBox("Saved recordings")
        form = QFormLayout(box)

        recs = backend.list_recordings()
        mb = backend.recordings_size() / (1024 * 1024)
        form.addRow(self._hint(
            f"Audio of every recording is kept in {backend.RECORDINGS_DIR} "
            f"(currently {len(recs)} files, {mb:.0f} MB)."))

        self.delete_on_start = QCheckBox("Delete all when the app starts")
        self.delete_on_start.setChecked(bool(self.cfg["delete_on_start"]))
        self.delete_on_start.setToolTip(backend.CONFIG_HELP["delete_on_start"])
        form.addRow(self.delete_on_start)

        self.auto_delete = QCheckBox("Delete old recordings while running")
        self.auto_delete.setChecked(bool(self.cfg["auto_delete"]))
        self.auto_delete.setToolTip(backend.CONFIG_HELP["auto_delete"])
        form.addRow(self.auto_delete)

        self.auto_hours = QSpinBox()
        self.auto_hours.setRange(1, 24 * 365)
        self.auto_hours.setSuffix(" hours")
        self.auto_hours.setValue(int(self.cfg["auto_delete_hours"]))
        self.auto_hours.setToolTip(backend.CONFIG_HELP["auto_delete_hours"])
        self.auto_hours.setEnabled(self.auto_delete.isChecked())
        self.auto_delete.toggled.connect(self.auto_hours.setEnabled)
        form.addRow("Older than", self.auto_hours)

        self.warn_mb = QSpinBox()
        self.warn_mb.setRange(0, 1_000_000)
        self.warn_mb.setSingleStep(256)
        self.warn_mb.setSuffix(" MB")
        self.warn_mb.setSpecialValueText("never")
        self.warn_mb.setValue(int(self.cfg["warn_size_mb"]))
        self.warn_mb.setToolTip(backend.CONFIG_HELP["warn_size_mb"])
        form.addRow("Warn above", self.warn_mb)
        return box

    def _hint(self, text: str) -> QLabel:
        label = QLabel(text)
        label.setWordWrap(True)
        pal = label.palette()
        pal.setColor(QPalette.ColorRole.WindowText,
                     pal.color(QPalette.ColorRole.PlaceholderText))
        label.setPalette(pal)
        return label

    # --- present/missing display -----------------------------------------

    def _refresh_radio(self, name: str) -> None:
        radio = self._radios[name]
        present = backend.model_present(name)
        radio.setText(f"{name}  ({'present' if present else 'missing'})")
        pal = radio.palette()
        role = (QPalette.ColorRole.WindowText if present
                else QPalette.ColorRole.PlaceholderText)
        pal.setColor(QPalette.ColorRole.WindowText, pal.color(role))
        radio.setPalette(pal)

    def _refresh_vad(self) -> None:
        present = backend.model_present("vad")
        self.vad_status.setText("(present)" if present else "(missing)")
        self.vad_btn.setVisible(not present)

    # --- model selection / download --------------------------------------

    def _selected(self) -> str:
        return self.group.checkedButton().property("model")

    def _on_model_clicked(self, button) -> None:
        name = button.property("model")
        if backend.model_present(name):
            self._current = name
            return
        entry = backend.MODEL_CATALOG.get(name)
        if entry is None:
            QMessageBox.warning(self, "Model missing",
                                f"{name} is not in llm_models/ and can't be downloaded.")
            self._radios[self._current].setChecked(True)
            return
        answer = QMessageBox.question(
            self, "Download model",
            f"{name} is not downloaded yet ({entry.size}).\nDownload it now?",
        )
        if answer == QMessageBox.StandardButton.Yes and self._start_download(name):
            self._current = name
        else:
            self._radios[self._current].setChecked(True)

    def _start_download(self, name: str) -> bool:
        if self._worker is not None:
            QMessageBox.information(self, "Download in progress",
                                    "Wait for the current download to finish.")
            return False
        self._worker = DownloadWorker(name)
        self._worker.progress.connect(self._on_progress)
        self._worker.done.connect(self._on_downloaded)
        self._worker.failed.connect(self._on_download_failed)
        self._worker.finished.connect(self._on_thread_finished)
        self.progress_label.setText(f"Downloading {name} …")
        self.progress_bar.setRange(0, 0)   # busy until the size is known
        self._show_progress(True)
        self._worker.start()
        return True

    def _on_progress(self, done: int, total: int) -> None:
        if total:
            self.progress_bar.setRange(0, 100)
            self.progress_bar.setValue(done * 100 // total)

    def _on_downloaded(self, name: str) -> None:
        if name in self._radios:
            self._refresh_radio(name)
        else:
            self._refresh_vad()

    def _on_download_failed(self, message: str) -> None:
        QMessageBox.critical(self, "Download failed", message)
        # A failed model download leaves the radio on a missing model; fall back.
        if not backend.model_present(self._current):
            for name in self._radios:
                if backend.model_present(name):
                    self._radios[name].setChecked(True)
                    self._current = name
                    break

    def _on_thread_finished(self) -> None:
        self._worker = None
        self._show_progress(False)
        self._refresh_vad()

    def _cancel_download(self) -> None:
        if self._worker:
            self._worker.request_cancel()

    def _show_progress(self, visible: bool) -> None:
        self.progress_label.setVisible(visible)
        self.progress_bar.setVisible(visible)
        self.cancel_btn.setVisible(visible)

    # --- save / close ------------------------------------------------------

    def _save(self) -> None:
        model = self._selected()
        if not backend.model_present(model):
            QMessageBox.warning(self, "Model missing",
                                "Download the selected model (or pick one that is "
                                "present) before saving.")
            return
        try:
            backend.save_config({
                "model": model,
                "threads": self.threads.value(),
                "language": self.language.text().strip(),
                "vad": self.vad.isChecked(),
                "timestamps": self.timestamps.isChecked(),
                "gpu": self.gpu_mode.currentData(),
                "gpu_device": (self.gpu_device.currentData()
                               if self.gpu_device.count() else self.cfg["gpu_device"]),
                "delete_on_start": self.delete_on_start.isChecked(),
                "auto_delete": self.auto_delete.isChecked(),
                "auto_delete_hours": self.auto_hours.value(),
                "warn_size_mb": self.warn_mb.value(),
            })
        except Exception as exc:  # noqa: BLE001
            QMessageBox.critical(self, "Could not save", str(exc))
            return
        self.accept()

    def done(self, result: int) -> None:
        # Never leave a download thread running behind a closed dialog.
        if self._worker is not None:
            self._worker.request_cancel()
            self._worker.wait()
        super().done(result)
