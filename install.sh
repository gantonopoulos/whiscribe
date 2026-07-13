#!/usr/bin/env bash
# Install whiscribe for the current user: CLI + tray on PATH, app icon, a
# Multimedia menu entry, and a systemd user service (start on login + journal).
set -euo pipefail

REPO="$(cd "$(dirname "$0")" && pwd)"

echo "Installing from $REPO"

# --- executables on PATH ---------------------------------------------------
mkdir -p ~/.local/bin
chmod +x "$REPO/whiscribe.py" "$REPO/tray.py"
ln -sf "$REPO/whiscribe.py" ~/.local/bin/whiscribe
ln -sf "$REPO/tray.py"      ~/.local/bin/whiscribe-tray

# --- application icon ------------------------------------------------------
ICONDIR=~/.local/share/icons/hicolor/scalable/apps
mkdir -p "$ICONDIR"
cp "$REPO/whiscribe.svg" "$ICONDIR/whiscribe.svg"

# --- menu entry (Multimedia) ----------------------------------------------
APPDIR=~/.local/share/applications
mkdir -p "$APPDIR"
# Reference the icon by absolute path so it renders even if the icon-theme
# cache hasn't picked up the newly added name.
sed "s|^Icon=whiscribe$|Icon=$ICONDIR/whiscribe.svg|" \
    "$REPO/whiscribe-tray.desktop" > "$APPDIR/whiscribe-tray.desktop"
command -v update-desktop-database >/dev/null 2>&1 && update-desktop-database "$APPDIR" || true
command -v kbuildsycoca6 >/dev/null 2>&1 && kbuildsycoca6 >/dev/null 2>&1 || true
command -v gtk-update-icon-cache  >/dev/null 2>&1 && gtk-update-icon-cache -f -t ~/.local/share/icons/hicolor 2>/dev/null || true

# --- systemd user service (autostart on login + journal logging) ----------
UNITDIR=~/.config/systemd/user
mkdir -p "$UNITDIR"
cp "$REPO/whiscribe-tray.service" "$UNITDIR/whiscribe-tray.service"
systemctl --user daemon-reload
systemctl --user enable whiscribe-tray.service

cat <<'EOF'

Done.
  Start now:      systemctl --user start whiscribe-tray
  (it also starts automatically on your next login)
  View logs:      journalctl --user -u whiscribe-tray -e
  Uninstall svc:  systemctl --user disable --now whiscribe-tray
EOF
