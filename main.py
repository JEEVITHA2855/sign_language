"""Main entrypoint for the AI Sign Language Recognition & Speech-to-Text System.

Usage:
    # 1. Launch Web Application (Default, Browser UI on http://localhost:8000)
    python main.py
    python main.py --web --port 8000

    # 2. Launch OpenCV Desktop Dashboard
    python main.py --opencv
    python main.py --opencv --mock --frames 60

    # 3. Run Automated Evaluation Benchmarks
    python main.py --eval all
    python main.py --eval isl
    python main.py --eval asl
    python main.py --eval speech
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


venv_py = PROJECT_ROOT / ".venv" / "Scripts" / "python.exe"
if venv_py.exists() and Path(sys.executable).resolve() != venv_py.resolve():
    import subprocess
    res = subprocess.run([str(venv_py)] + sys.argv, check=False)
    sys.exit(res.returncode)


def launch_web(host: str = "127.0.0.1", port: int = 8000) -> None:
    """Launch the Web API server and frontend interface."""
    from src.api.server import run_server
    run_server(host=host, port=port)


def launch_opencv(
    config_path: str = "config/config.yaml",
    camera_idx: int | None = None,
    mock: bool = False,
    no_tts: bool = False,
    mode: str | None = None,
    frames: int | None = None,
) -> None:
    """Launch the OpenCV HUD window application."""
    from src.app import SignLanguageApp
    app = SignLanguageApp(
        config_path=config_path,
        camera_override=camera_idx,
        mock_mode=mock,
        disable_tts=no_tts,
        mode_override=mode,
    )
    app.run(max_frames=frames)


def run_evaluations(eval_target: str) -> None:
    """Run specified evaluation benchmarks."""
    import subprocess

    python_bin = sys.executable
    targets = ["isl", "asl", "speech"] if eval_target == "all" else [eval_target]

    for target in targets:
        print(f"\n{'=' * 60}")
        print(f"  RUNNING {target.upper()} EVALUATION BENCHMARK")
        print(f"{'=' * 60}")
        if target == "isl":
            subprocess.run([python_bin, "evaluation/evaluate_isl.py", "--benchmark-only"])
        elif target == "asl":
            subprocess.run([python_bin, "evaluation/evaluate_asl.py", "--mode", "BOTH"])
        elif target == "speech":
            subprocess.run([python_bin, "evaluation/evaluate_speech.py", "--benchmark-only"])


def main() -> None:
    parser = argparse.ArgumentParser(
        description="OmniSign AI: Multimodal Sign Language & Speech-to-Text System",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--opencv", action="store_true", help="Launch OpenCV desktop GUI application (Default)")
    group.add_argument("--web", action="store_true", help="Launch Web application server")
    group.add_argument("--eval", choices=["isl", "asl", "speech", "all"], help="Run evaluation benchmarks")

    # Web options
    parser.add_argument("--host", type=str, default="127.0.0.1", help="Host address for web server")
    parser.add_argument("--port", type=int, default=8000, help="Port for web server")

    # OpenCV options
    parser.add_argument("--config", type=str, default="config/config.yaml", help="Path to config.yaml")
    parser.add_argument("--camera", type=int, default=None, help="Camera device index")
    parser.add_argument("--mock", action="store_true", help="Run with mock camera frames")
    parser.add_argument("--no-tts", action="store_true", help="Disable voice audio synthesis")
    parser.add_argument("--mode", choices=["WORD", "FINGERSPELLING", "AUTO"], default=None, help="Pipeline mode")
    parser.add_argument("--frames", type=int, default=None, help="Exit after N frames (testing)")

    args = parser.parse_args()

    if args.web:
        launch_web(host=args.host, port=args.port)
    elif args.eval:
        run_evaluations(args.eval)
    else:
        # Default: Launch OpenCV Desktop Camera Application
        launch_opencv(
            config_path=args.config,
            camera_idx=args.camera,
            mock=args.mock,
            no_tts=args.no_tts,
            mode=args.mode,
            frames=args.frames,
        )


if __name__ == "__main__":
    main()
