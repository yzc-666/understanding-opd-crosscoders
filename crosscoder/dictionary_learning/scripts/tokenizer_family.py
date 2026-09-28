"""Print the vocabulary fingerprint of a model or of an existing token stream."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from transformers import AutoTokenizer

from dictionary_learning.scripts.prepare_opd_crosscoder_tokens import vocab_fingerprint

SHORT_LENGTH = 12


def model_fingerprint(model_path: str) -> str:
    return vocab_fingerprint(AutoTokenizer.from_pretrained(model_path, trust_remote_code=True))


def token_dir_fingerprint(token_dir: str) -> str:
    manifest = json.loads((Path(token_dir) / "manifest.json").read_text())
    return manifest.get("tokenizer_vocab_sha256") or model_fingerprint(manifest["tokenizer"])


def main() -> None:
    parser = argparse.ArgumentParser()
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--model")
    group.add_argument("--token-dir")
    parser.add_argument("--short", action="store_true")
    args = parser.parse_args()
    digest = model_fingerprint(args.model) if args.model else token_dir_fingerprint(args.token_dir)
    print(digest[:SHORT_LENGTH] if args.short else digest)


if __name__ == "__main__":
    main()
