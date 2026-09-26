import os
import sys
import getpass
import subprocess
import webbrowser
from pathlib import Path

# Base project directory
BASE_DIR = Path(__file__).resolve().parent

# Define virtual environment path
VENV_DIR = BASE_DIR / "venv"
IS_WINDOWS = sys.platform == "win32"

# Paths to python and pip executables inside the virtual environment
if IS_WINDOWS:
    PYTHON_BIN = VENV_DIR / "Scripts" / "python.exe"
    PIP_BIN = VENV_DIR / "Scripts" / "pip.exe"
else:
    PYTHON_BIN = VENV_DIR / "bin" / "python"
    PIP_BIN = VENV_DIR / "bin" / "pip"


def prompt_hf_token():
    """Prompt the user for their Hugging Face token and set it in environment variables."""
    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    
    if not token:
        print("\n" + "=" * 60)
        print("Hugging Face authentication required (pyannote.audio)")
        print("=" * 60)
        token = getpass.getpass("Enter your HF Token (input hidden): ").strip()

        if not token:
            print("[WARNING] No token entered. pyannote.audio may fail if models require authentication.")
        else:
            print("[INFO] Hugging Face token captured successfully.")

    env = os.environ.copy()
    if token:
        env["HF_TOKEN"] = token
        env["HUGGING_FACE_HUB_TOKEN"] = token
    
    return env


def run_command(command, cwd=BASE_DIR, env=None):
    """Utility to run shell commands safely and handle errors."""
    print(f"\n[EXEC] Running: {' '.join(command) if isinstance(command, list) else command}")
    try:
        subprocess.run(command, cwd=cwd, env=env, check=True, shell=isinstance(command, str))
    except subprocess.CalledProcessError as e:
        print(f"\n[ERROR] Command failed with exit code {e.returncode}: {command}")
        sys.exit(1)


def setup_virtualenv(env):
    """Create virtual environment if it doesn't exist and install dependencies."""
    if not VENV_DIR.exists():
        print("[INFO] Creating virtual environment...")
        run_command([sys.executable, "-m", "venv", str(VENV_DIR)], env=env)
    else:
        print("[INFO] Virtual environment already exists.")

    requirements_file = BASE_DIR / "requirements.txt"
    if requirements_file.exists():
        print("[INFO] Installing dependencies from requirements.txt...")
        run_command([str(PIP_BIN), "install", "-r", str(requirements_file)], env=env)
    else:
        print("[WARNING] requirements.txt not found. Skipping dependency installation.")


def setup_docker(env):
    """Start PostgreSQL container using Docker Compose."""
    db_setup_dir = BASE_DIR / "db_setup"
    docker_file = db_setup_dir / "postgress_setup.yaml"

    if not docker_file.exists():
        print(f"[ERROR] Docker Compose file not found at: {docker_file}")
        sys.exit(1)

    print("[INFO] Starting PostgreSQL with Docker Compose...")
    try:
        run_command(["docker", "compose", "-f", "postgress_setup.yaml", "up", "-d"], cwd=db_setup_dir, env=env)
    except SystemExit:
        print("[INFO] Retrying with 'docker-compose'...")
        run_command(["docker-compose", "-f", "postgress_setup.yaml", "up", "-d"], cwd=db_setup_dir, env=env)


def should_skip_step(step_name):
    """Checks directory structure to determine if step outputs already exist."""
    data_dir = BASE_DIR / "data"
    
    if step_name == "transcripts_diarization.py":
        diarized_dir = data_dir / "transcript_diarized"
        # Check if JSON files exist in transcript_diarized directory
        if diarized_dir.exists() and any(diarized_dir.glob("*.json")):
            return True

    elif step_name == "transcript_chunking.py":
        chunked_dir = data_dir / "transcript_diarized" / "chunked"
        # Check if JSON files exist inside transcript_diarized/chunked directory
        if chunked_dir.exists() and any(chunked_dir.glob("*.json")):
            return True

    return False


def run_pipeline(env):
    """Execute Python pipeline scripts in sequential order, skipping completed steps."""
    pipeline_scripts = [
        "transripts_diarization.py",
        "transcript_chunking.py",
        "ingest.py",
        "eval_recall.py"
    ]

    for script in pipeline_scripts:
        script_path = BASE_DIR / script
        if not script_path.exists():
            print(f"[ERROR] Script not found: {script_path}")
            sys.exit(1)

        if should_skip_step(script):
            print(f"\n[SKIP] Output files already exist for {script}. Skipping step.")
            continue

        print(f"\n[INFO] Executing {script}...")
        run_command([str(PYTHON_BIN), str(script_path)], cwd=BASE_DIR, env=env)


def open_report():
    """Open eval_report.html in default browser upon completion."""
    report_path = BASE_DIR / "eval_report.html"
    if report_path.exists():
        print("\n[SUCCESS] Pipeline execution complete. Opening evaluation report...")
        webbrowser.open(report_path.as_uri())
    else:
        print(f"\n[WARNING] Report generated file not found at: {report_path}")


def main():
    env = prompt_hf_token()
    setup_virtualenv(env)
    setup_docker(env)
    run_pipeline(env)
    open_report()


if __name__ == "__main__":
    main()