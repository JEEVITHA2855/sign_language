from __future__ import annotations

import argparse
from pathlib import Path

from huggingface_hub import snapshot_download


def download_asl() -> None:
    dest = Path('models/asl')
    dest.mkdir(parents=True, exist_ok=True)
    snapshot_download('raghuhasan/asl2000-i3d', local_dir=str(dest), local_dir_use_symlinks=False)
    print('ASL model downloaded to', dest)


def download_isl() -> None:
    dest = Path('models/isl')
    dest.mkdir(parents=True, exist_ok=True)
    snapshot_download('Abs6187/isl-models', local_dir=str(dest), local_dir_use_symlinks=False)
    print('ISL model downloaded to', dest)


def main() -> None:
    parser = argparse.ArgumentParser(description='Download local word-level ASL/ISL model assets.')
    parser.add_argument('--asl', action='store_true', help='Download ASL model weights.')
    parser.add_argument('--isl', action='store_true', help='Download ISL model weights.')
    args = parser.parse_args()

    if args.asl:
        download_asl()
    if args.isl:
        download_isl()
    if not args.asl and not args.isl:
        print('Specify --asl and/or --isl.')


if __name__ == '__main__':
    main()
