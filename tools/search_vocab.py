"""Vocabulary Search & Inspection Tool for OmniSign AI.

Allows searching and listing the 2,000+ words recognized by the system.

Usage:
    python tools/search_vocab.py --query doctor
    python tools/search_vocab.py --sample 50
    python tools/search_vocab.py --language ISL
    python tools/search_vocab.py --language ASL
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description="Inspect sign language vocabulary")
    parser.add_argument("--query", "-q", type=str, default=None, help="Word to search for")
    parser.add_argument("--sample", "-s", type=int, default=30, help="Number of random samples to show")
    parser.add_argument("--language", "-l", choices=["ASL", "ISL", "ALL"], default="ALL", help="Language vocabulary")
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parent.parent

    # ASL Vocabulary (2,000 words from WLASL)
    asl_labels_file = project_root / "models" / "asl" / "labels.json"
    asl_words = []
    if asl_labels_file.exists():
        with open(asl_labels_file, "r", encoding="utf-8") as f:
            asl_words = json.load(f)

    # ISL Vocabulary
    isl_meta_file = project_root / "models" / "isl" / "metadata.json"
    isl_words = []
    if isl_meta_file.exists():
        with open(isl_meta_file, "r", encoding="utf-8") as f:
            meta = json.load(f)
            isl_words = [c.strip() for c in meta.get("class_names", [])]

    print("\n" + "=" * 65)
    print("  OMNISIGN AI - VOCABULARY MATRIX & INSPECTOR")
    print("=" * 65)
    print(f"  ASL Word Model (WLASL-I3D)  : {len(asl_words):,} words loaded")
    print(f"  ISL Word Model (Hierarchical): {len(isl_words):,} words loaded")
    print(f"  Alphabet Fingerspelling     : Unlimited (A-Z spells ANY word)")
    print("=" * 65)

    if args.query:
        q = args.query.lower().strip()
        print(f"\nSearching for '{q}':\n")
        asl_matches = [w for w in asl_words if q in w.lower()]
        isl_matches = [w for w in isl_words if q in w.lower()]
        print(f"  ASL Matches ({len(asl_matches)}):")
        for m in asl_matches[:20]:
            print(f"    - {m}")
        if len(asl_matches) > 20:
            print(f"    ... and {len(asl_matches) - 20} more.")

        print(f"\n  ISL Matches ({len(isl_matches)}):")
        for m in isl_matches:
            print(f"    - {m}")
    else:
        print(f"\nSample of {args.sample} advanced/diverse ASL words (out of {len(asl_words):,} total):")
        step = max(1, len(asl_words) // args.sample)
        sample = asl_words[::step][:args.sample]
        for i, word in enumerate(sample, 1):
            print(f"  {i:>2}. {word:<18}", end="\n" if i % 3 == 0 else "")
        print("\n")


if __name__ == "__main__":
    main()
