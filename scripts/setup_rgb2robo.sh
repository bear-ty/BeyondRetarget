#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
ENV_NAME="BeyondRetarget"
DRIVE_FOLDER_URL="https://drive.google.com/drive/folders/12ySWLbVhF8WEGyLR0Ll09DmZm8rZE0Tg"
YOLOV8X_FILE_ID="${RGB2ROBO_YOLOV8X_FILE_ID:-}"
CHECK_ONLY=false
SKIP_ENV=false
REUSE_ENV=false
SKIP_DOWNLOAD=false
FORCE_DOWNLOAD=false
ALLOW_NO_CUDA=false
QUICK_CHECK=false

usage() {
    printf '%s\n' \
        "RGB2Robo one-command setup." \
        "" \
        "Usage: bash scripts/setup_rgb2robo.sh [options]" \
        "" \
        "  --env-name NAME          Conda environment name (default: BeyondRetarget)." \
        "  --drive-folder-url URL   Public Google Drive resource folder." \
        "  --yolov8x-file-id ID     Public Drive file ID if folder listing is delayed." \
        "  --check-only             Do not create/update the environment or download files." \
        "  --skip-env               Use the active Python environment without modifying Conda." \
        "  --reuse-env              Reuse an existing Conda environment without updating it." \
        "  --skip-download          Do not download assets; only validate existing files." \
        "  --force-download         Download all external assets again." \
        "  --allow-no-cuda          Report missing CUDA as a warning instead of an error." \
        "  --quick-check            Skip model deserialization and ONNX Runtime graph loading." \
        "  -h, --help               Show this help."
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --env-name)
            ENV_NAME="$2"
            shift 2
            ;;
        --drive-folder-url)
            DRIVE_FOLDER_URL="$2"
            shift 2
            ;;
        --yolov8x-file-id)
            YOLOV8X_FILE_ID="$2"
            shift 2
            ;;
        --check-only)
            CHECK_ONLY=true
            shift
            ;;
        --skip-env)
            SKIP_ENV=true
            shift
            ;;
        --reuse-env)
            REUSE_ENV=true
            shift
            ;;
        --skip-download)
            SKIP_DOWNLOAD=true
            shift
            ;;
        --force-download)
            FORCE_DOWNLOAD=true
            shift
            ;;
        --allow-no-cuda)
            ALLOW_NO_CUDA=true
            shift
            ;;
        --quick-check)
            QUICK_CHECK=true
            shift
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        *)
            echo "Unknown option: $1" >&2
            usage >&2
            exit 2
            ;;
    esac
done

cd "$PROJECT_ROOT"

if [[ "$SKIP_ENV" == true ]]; then
    PYTHON_BIN="${PYTHON_BIN:-$(command -v python3 || true)}"
    if [[ -z "$PYTHON_BIN" ]]; then
        echo "python3 was not found in the active environment." >&2
        exit 1
    fi
    run_python() {
        "$PYTHON_BIN" "$@"
    }
else
    CONDA_BIN="${CONDA_EXE:-$(command -v conda || true)}"
    if [[ -z "$CONDA_BIN" ]]; then
        echo "Conda was not found. Install Miniconda/Anaconda or use --skip-env." >&2
        exit 1
    fi

    if ! "$CONDA_BIN" run -n "$ENV_NAME" python -c "import sys; print(sys.executable)" >/dev/null 2>&1; then
        if [[ "$CHECK_ONLY" == true ]]; then
            echo "Conda environment '$ENV_NAME' does not exist; run setup without --check-only." >&2
            exit 1
        fi
        echo "[env] creating Conda environment '$ENV_NAME'"
        "$CONDA_BIN" env create -n "$ENV_NAME" -f environment.yml
    elif [[ "$CHECK_ONLY" == false && "$REUSE_ENV" == false ]]; then
        if ! "$CONDA_BIN" run -n "$ENV_NAME" python -m pip --version >/dev/null 2>&1; then
            echo "[env] repairing pip in '$ENV_NAME'"
            "$CONDA_BIN" install -n "$ENV_NAME" --force-reinstall -y pip
        fi
        echo "[env] updating '$ENV_NAME' from environment.yml"
        "$CONDA_BIN" env update -n "$ENV_NAME" -f environment.yml
    else
        echo "[env] reusing Conda environment '$ENV_NAME'"
    fi

    run_python() {
        "$CONDA_BIN" run -n "$ENV_NAME" python "$@"
    }
fi

if [[ "$CHECK_ONLY" == false && "$SKIP_DOWNLOAD" == false ]]; then
    download_args=(
        scripts/download_rgb2robo_assets.py
        --project-root "$PROJECT_ROOT"
        --folder-url "$DRIVE_FOLDER_URL"
    )
    if [[ -n "$YOLOV8X_FILE_ID" ]]; then
        download_args+=(--yolov8x-file-id "$YOLOV8X_FILE_ID")
    fi
    if [[ "$FORCE_DOWNLOAD" == true ]]; then
        download_args+=(--force)
    fi
    run_python "${download_args[@]}"
fi

check_args=(scripts/check_installation.py --project-root "$PROJECT_ROOT")
if [[ "$ALLOW_NO_CUDA" == true ]]; then
    check_args+=(--allow-no-cuda)
fi
if [[ "$QUICK_CHECK" == true ]]; then
    check_args+=(--quick)
fi

echo "[check] validating the RGB2Robo installation"
run_python "${check_args[@]}"

echo
echo "Setup completed. Activate the environment with: conda activate $ENV_NAME"
