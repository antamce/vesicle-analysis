#!/bin/bash

set -u

PROJECT_ROOT="$(cd "$(dirname "$0")" && pwd)"
ENVIRONMENT_FILE="$PROJECT_ROOT/environment.yml"
MAC_ICON="$PROJECT_ROOT/launcher/assets/icon.icns"
CONFIG_DIRECTORY="$HOME/Library/Application Support/VesicleAnalysis"
CONDA_CONFIG="$CONFIG_DIRECTORY/conda_path"
ENVIRONMENT_CONFIG="$CONFIG_DIRECTORY/environment_name"
HASH_CONFIG="$CONFIG_DIRECTORY/environment_hash"
MARKER_NAME="vesicle-analysis-launcher.marker"
MARKER_VALUE="vesicle-analysis-project-environment-v1"

write_step() {
    printf '\033[36m[Vesicle Analysis] %s\033[0m\n' "$1"
}

fail() {
    printf '\n\033[31mVesicle Analysis launcher error:\n%s\033[0m\n\n' "$1"
    printf 'If Conda is not installed, install Anaconda, Miniconda, Miniforge, or Mambaforge and try again.\n'
    read -r -p "Press Return to close this window..." _unused
    exit 1
}

resolve_conda_candidate() {
    candidate="${1:-}"
    [ -n "$candidate" ] || return 1
    if [ -d "$candidate" ]; then
        for relative in bin/conda condabin/conda; do
            executable="$candidate/$relative"
            if [ -x "$executable" ] && "$executable" --version 2>/dev/null | grep -qi conda; then
                printf '%s\n' "$executable"
                return 0
            fi
        done
    elif [ -x "$candidate" ] && "$candidate" --version 2>/dev/null | grep -qi conda; then
        printf '%s\n' "$candidate"
        return 0
    fi
    return 1
}

find_conda_automatically() {
    candidates=""
    if [ -f "$CONDA_CONFIG" ]; then
        candidates="$(cat "$CONDA_CONFIG")"
    fi
    if [ -n "${CONDA_EXE:-}" ]; then
        candidates="$candidates
$CONDA_EXE"
    fi
    command_path="$(command -v conda 2>/dev/null || true)"
    if [ -n "$command_path" ]; then
        candidates="$candidates
$command_path"
    fi
    candidates="$candidates
$HOME/anaconda3
$HOME/miniconda3
$HOME/miniforge3
$HOME/mambaforge
/opt/anaconda3
/opt/miniconda3
/opt/miniforge3
/opt/homebrew
/usr/local/anaconda3
/usr/local/miniconda3
/usr/local/miniforge3"
    while IFS= read -r candidate; do
        [ -n "$candidate" ] || continue
        resolved="$(resolve_conda_candidate "$candidate" || true)"
        if [ -n "$resolved" ]; then
            printf '%s\n' "$resolved"
            return 0
        fi
    done <<EOF
$candidates
EOF
    return 1
}

request_conda_path() {
    while true; do
        choice="$(osascript -e 'button returned of (display dialog "Conda was not found automatically. Choose the Conda executable or its Anaconda, Miniconda, Miniforge, or Mambaforge installation folder. If Conda is not installed, install one of those distributions and run this launcher again." with title "Locate Conda" buttons {"Cancel", "Choose folder", "Choose executable"} default button "Choose executable" with icon note)' 2>/dev/null || true)"
        case "$choice" in
            "Choose executable")
                selected="$(osascript -e 'POSIX path of (choose file with prompt "Select the Conda executable")' 2>/dev/null || true)"
                ;;
            "Choose folder")
                selected="$(osascript -e 'POSIX path of (choose folder with prompt "Select the Conda installation folder")' 2>/dev/null || true)"
                selected="${selected%/}"
                ;;
            *)
                return 1
                ;;
        esac
        [ -n "$selected" ] || continue
        resolved="$(resolve_conda_candidate "$selected" || true)"
        if [ -n "$resolved" ]; then
            printf '%s\n' "$resolved"
            return 0
        fi
        osascript -e 'display alert "Invalid Conda path" message "That path does not contain a working Conda executable. Select the executable or top-level installation folder and try again." as warning' >/dev/null 2>&1 || true
    done
}

environment_prefix() {
    "$1" run --name "$2" python -c 'import sys; print(sys.prefix)' 2>/dev/null | tail -n 1
}

is_managed_environment() {
    prefix="$1"
    [ -n "$prefix" ] && [ -f "$prefix/conda-meta/$MARKER_NAME" ] &&
        [ "$(cat "$prefix/conda-meta/$MARKER_NAME")" = "$MARKER_VALUE" ]
}

select_environment_name() {
    conda_path="$1"
    if [ -f "$ENVIRONMENT_CONFIG" ]; then
        saved_name="$(cat "$ENVIRONMENT_CONFIG")"
        saved_prefix="$(environment_prefix "$conda_path" "$saved_name")"
        if is_managed_environment "$saved_prefix"; then
            printf '%s\n' "$saved_name"
            return 0
        fi
    fi
    index=0
    while [ "$index" -lt 100 ]; do
        if [ "$index" -eq 0 ]; then
            name="vesicle-analysis"
        elif [ "$index" -eq 1 ]; then
            name="vesicle-analysis-app"
        else
            name="vesicle-analysis-app-$index"
        fi
        prefix="$(environment_prefix "$conda_path" "$name")"
        if [ -z "$prefix" ] || is_managed_environment "$prefix"; then
            printf '%s\n' "$name"
            return 0
        fi
        index=$((index + 1))
    done
    return 1
}

ensure_environment() {
    conda_path="$1"
    environment_name="$2"
    saved_hash="$3"
    current_hash="$4"
    prefix="$(environment_prefix "$conda_path" "$environment_name")"
    if [ -z "$prefix" ]; then
        write_step "Creating the dedicated '$environment_name' environment. Internet access may be required." >&2
        "$conda_path" env create --name "$environment_name" --file "$ENVIRONMENT_FILE" --yes 1>&2 ||
            return 1
        prefix="$(environment_prefix "$conda_path" "$environment_name")"
        [ -n "$prefix" ] || return 1
        printf '%s\n' "$MARKER_VALUE" > "$prefix/conda-meta/$MARKER_NAME" || return 1
    elif ! is_managed_environment "$prefix"; then
        return 1
    elif [ "$saved_hash" != "$current_hash" ]; then
        write_step "Updating '$environment_name' because environment.yml changed. Internet access may be required." >&2
        "$conda_path" env update --name "$environment_name" --file "$ENVIRONMENT_FILE" --prune --yes 1>&2 ||
            return 1
    fi
    printf '%s\n' "$prefix"
}

create_desktop_shortcut() {
    desktop="$(osascript -e 'POSIX path of (path to desktop folder)' 2>/dev/null || true)"
    desktop="${desktop%/}"
    [ -n "$desktop" ] || return 1
    app="$desktop/Vesicle Analysis.app"
    marker="$app/Contents/Resources/vesicle-analysis-shortcut.marker"
    if [ -e "$app" ]; then
        if [ ! -f "$marker" ] || [ "$(cat "$marker")" != "$MARKER_VALUE" ]; then
            return 1
        fi
        rm -rf -- "$app" || return 1
    fi
    source_file="$CONFIG_DIRECTORY/desktop-launcher.applescript"
    escaped_launcher="$(printf '%s' "$PROJECT_ROOT/Launch Vesicle Analysis.command" | sed 's/\\/\\\\/g; s/"/\\"/g')"
    cat > "$source_file" <<EOF
property launcherPath : "$escaped_launcher"

on run
    launchVesicleAnalysis({})
end run

on open droppedItems
    launchVesicleAnalysis(droppedItems)
end open

on launchVesicleAnalysis(droppedItems)
    set shellCommand to quoted form of launcherPath
    repeat with droppedItem in droppedItems
        set shellCommand to shellCommand & " " & quoted form of POSIX path of droppedItem
    end repeat
    tell application "Terminal"
        activate
        do script shellCommand
    end tell
end launchVesicleAnalysis
EOF
    osacompile -o "$app" "$source_file" || return 1
    mkdir -p "$app/Contents/Resources" || return 1
    printf '%s\n' "$MARKER_VALUE" > "$marker" || return 1
    if [ -f "$MAC_ICON" ]; then
        cp "$MAC_ICON" "$app/Contents/Resources/applet.icns"
    fi
    touch "$app"
    write_step "Desktop shortcut ready: $app"
}

close_terminal_after_success() {
    current_tty="$(tty 2>/dev/null || true)"
    case "$current_tty" in
        /dev/*)
            nohup osascript - "$current_tty" >/dev/null 2>&1 <<'APPLESCRIPT' &
on run argv
    delay 0.5
    tell application "Terminal"
        repeat with currentWindow in windows
            repeat with currentTab in tabs of currentWindow
                if tty of currentTab is item 1 of argv then
                    close currentWindow
                    return
                end if
            end repeat
        end repeat
    end tell
end run
APPLESCRIPT
            ;;
    esac
}

[ -f "$ENVIRONMENT_FILE" ] || fail "environment.yml is missing from $PROJECT_ROOT."
mkdir -p "$CONFIG_DIRECTORY" || fail "Could not create the per-user launcher configuration folder."

CONDA_PATH="$(find_conda_automatically || true)"
if [ -z "$CONDA_PATH" ]; then
    CONDA_PATH="$(request_conda_path || true)"
fi
[ -n "$CONDA_PATH" ] || fail "Conda was not selected."
write_step "Using Conda: $CONDA_PATH"
printf '%s\n' "$CONDA_PATH" > "$CONDA_CONFIG"

ENVIRONMENT_NAME="$(select_environment_name "$CONDA_PATH" || true)"
[ -n "$ENVIRONMENT_NAME" ] || fail "Could not choose an unused managed environment name."
CURRENT_HASH="$(shasum -a 256 "$ENVIRONMENT_FILE" | awk '{print $1}')"
SAVED_HASH=""
if [ -f "$ENVIRONMENT_CONFIG" ] && [ "$(cat "$ENVIRONMENT_CONFIG")" = "$ENVIRONMENT_NAME" ] && [ -f "$HASH_CONFIG" ]; then
    SAVED_HASH="$(cat "$HASH_CONFIG")"
fi
PREFIX="$(ensure_environment "$CONDA_PATH" "$ENVIRONMENT_NAME" "$SAVED_HASH" "$CURRENT_HASH")" ||
    fail "Conda could not create or update the dedicated environment. Check the internet connection and messages above, then try again."
is_managed_environment "$PREFIX" || fail "The managed-environment ownership marker could not be verified."

printf '%s\n' "$CONDA_PATH" > "$CONDA_CONFIG"
printf '%s\n' "$ENVIRONMENT_NAME" > "$ENVIRONMENT_CONFIG"
printf '%s\n' "$CURRENT_HASH" > "$HASH_CONFIG"
create_desktop_shortcut || fail "The macOS desktop shortcut could not be created."

write_step "Starting Vesicle Analysis. Application output is saved under $CONFIG_DIRECTORY."
nohup "$CONDA_PATH" run --no-capture-output --name "$ENVIRONMENT_NAME" \
    python -m vesicle_analysis "$@" \
    > "$CONFIG_DIRECTORY/app-output.log" \
    2> "$CONFIG_DIRECTORY/app-error.log" < /dev/null &
APP_PID=$!
sleep 2
if ! kill -0 "$APP_PID" 2>/dev/null; then
    wait "$APP_PID"
    APP_STATUS=$?
    if [ "$APP_STATUS" -ne 0 ]; then
        ERROR_DETAILS="$(tail -n 20 "$CONFIG_DIRECTORY/app-error.log" 2>/dev/null || true)"
        fail "The application exited during startup.
$ERROR_DETAILS"
    fi
fi
close_terminal_after_success
exit 0
