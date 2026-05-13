#!/usr/bin/env bash
# build_app.sh — Builds GlobalWatch.app (double-click launcher for macOS)
# Run once: bash build_app.sh
# After that, double-click GlobalWatch.app to start the GUI.
# Re-run any time you move the project folder.

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
APP="$SCRIPT_DIR/GlobalWatch.app"
DESKTOP_SCRIPT="$SCRIPT_DIR/GlobalWatch_Desktop.py"

echo "Building GlobalWatch.app ..."

# ── Sanity checks ─────────────────────────────────────────────────────────────
[ -f "$DESKTOP_SCRIPT" ] || { echo "ERROR: GlobalWatch_Desktop.py not found"; exit 1; }
which swiftc >/dev/null 2>&1 || { echo "ERROR: swiftc not found. Install Xcode Command Line Tools: xcode-select --install"; exit 1; }
PYTHON_BIN=$(command -v python3 || true)
[ -n "$PYTHON_BIN" ] || { echo "ERROR: python3 not found"; exit 1; }

# ── Install Python dependencies if missing ────────────────────────────────────
"$PYTHON_BIN" -c "import customtkinter, matplotlib" 2>/dev/null || {
    echo "Installing customtkinter and matplotlib ..."
    "$PYTHON_BIN" -m pip install --quiet customtkinter matplotlib
}

# ── Compile Swift launcher ────────────────────────────────────────────────────
SWIFT_SRC=$(mktemp /tmp/GWLauncher.XXXXXX.swift)
trap "rm -f $SWIFT_SRC" EXIT

cat > "$SWIFT_SRC" << 'SWIFT'
import Foundation
import AppKit

// Framework Python is required for GUI rendering from a .app bundle on macOS.
// Non-framework python3 shows a black/empty window when launched from Finder.
func findPython() -> String? {
    let candidates = [
        "/Library/Developer/CommandLineTools/Library/Frameworks/Python3.framework/Versions/3.9/Resources/Python.app/Contents/MacOS/Python",
        "/Library/Developer/CommandLineTools/Library/Frameworks/Python3.framework/Versions/3.11/Resources/Python.app/Contents/MacOS/Python",
        "/Library/Developer/CommandLineTools/Library/Frameworks/Python3.framework/Versions/3.12/Resources/Python.app/Contents/MacOS/Python",
        "/opt/homebrew/opt/python@3.12/Frameworks/Python.framework/Versions/3.12/Resources/Python.app/Contents/MacOS/Python",
        "/opt/homebrew/opt/python@3.11/Frameworks/Python.framework/Versions/3.11/Resources/Python.app/Contents/MacOS/Python",
        "/Library/Frameworks/Python.framework/Versions/3.12/Resources/Python.app/Contents/MacOS/Python",
        "/Library/Frameworks/Python.framework/Versions/3.11/Resources/Python.app/Contents/MacOS/Python",
        "/Library/Frameworks/Python.framework/Versions/3.9/Resources/Python.app/Contents/MacOS/Python",
    ]
    return candidates.first { FileManager.default.isExecutableFile(atPath: $0) }
}

let appURL      = Bundle.main.bundleURL
let projectDir  = appURL.deletingLastPathComponent().path
let scriptPath  = "\(projectDir)/GlobalWatch_Desktop.py"

guard let python = findPython() else {
    let a = NSAlert()
    a.messageText = "Python 3 not found"
    a.informativeText = "Install Python 3 from https://www.python.org/ and rebuild the app."
    a.alertStyle = .critical; a.runModal(); exit(1)
}

guard FileManager.default.fileExists(atPath: scriptPath) else {
    let a = NSAlert()
    a.messageText = "GlobalWatch_Desktop.py not found"
    a.informativeText = "Expected at: \(scriptPath)\nPlease rebuild the app from the correct folder."
    a.alertStyle = .critical; a.runModal(); exit(1)
}

let task = Process()
task.executableURL = URL(fileURLWithPath: python)
task.arguments    = [scriptPath]
task.currentDirectoryURL = URL(fileURLWithPath: projectDir)
var env = ProcessInfo.processInfo.environment
env["PYTHONPATH"] = projectDir
task.environment = env

do {
    try task.run()
    task.waitUntilExit()
} catch {
    let a = NSAlert()
    a.messageText = "Launch failed"
    a.informativeText = error.localizedDescription
    a.alertStyle = .critical; a.runModal(); exit(1)
}
SWIFT

LAUNCHER_BIN=$(mktemp /tmp/GlobalWatchLauncher.XXXXXX)
trap "rm -f $SWIFT_SRC $LAUNCHER_BIN" EXIT

swiftc "$SWIFT_SRC" \
    -framework Foundation \
    -framework AppKit \
    -o "$LAUNCHER_BIN" \
    -target arm64-apple-macosx11.0 \
    2>&1

# ── Assemble .app bundle ──────────────────────────────────────────────────────
rm -rf "$APP"
mkdir -p "$APP/Contents/MacOS" "$APP/Contents/Resources"

cp "$LAUNCHER_BIN" "$APP/Contents/MacOS/GlobalWatch"
chmod +x "$APP/Contents/MacOS/GlobalWatch"

cat > "$APP/Contents/Info.plist" << 'PLIST'
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
    <key>CFBundleName</key><string>GlobalWatch</string>
    <key>CFBundleDisplayName</key><string>GlobalWatch</string>
    <key>CFBundleIdentifier</key><string>com.globalwatch.desktop</string>
    <key>CFBundleVersion</key><string>2.0</string>
    <key>CFBundleShortVersionString</key><string>2.0</string>
    <key>CFBundleExecutable</key><string>GlobalWatch</string>
    <key>CFBundlePackageType</key><string>APPL</string>
    <key>NSHighResolutionCapable</key><true/>
    <key>NSRequiresAquaSystemAppearance</key><false/>
    <key>LSMinimumSystemVersion</key><string>11.0</string>
</dict>
</plist>
PLIST

# Remove quarantine so macOS doesn't block it
xattr -rd com.apple.quarantine "$APP" 2>/dev/null || true

echo ""
echo "✓  Built: $APP"
echo ""
echo "Usage:"
echo "  Double-click GlobalWatch.app in Finder to launch"
echo "  Or: open \"$APP\""
echo ""
echo "If macOS shows a security warning:"
echo "  System Settings → Privacy & Security → scroll down → Open Anyway"
echo ""
echo "Note: Re-run build_app.sh if you move the project folder."
