"""Build the fixed token stream shared by every model of a crosscoder."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import shutil
from collections.abc import Iterable, Iterator
from pathlib import Path

import torch
from datasets import load_dataset
from transformers import AutoTokenizer


DEFAULT_REASONING = "open-thoughts/OpenThoughts-114k"
DEFAULT_GENERAL = "togethercomputer/RedPajama-Data-1T-Sample"


def vocab_fingerprint(tokenizer) -> str:
    """Hash the full vocabulary: families of the same size can still map text to different ids."""

    items = sorted(tokenizer.get_vocab().items())
    payload = json.dumps(items, ensure_ascii=False).encode()
    return hashlib.sha256(payload).hexdigest()


def _atomic_json(path: Path, value: dict) -> None:
    temporary = path.with_suffix(f"{path.suffix}.partial")
    with temporary.open("w") as handle:
        json.dump(value, handle, indent=2, sort_keys=True)
    os.replace(temporary, path)


PROMPT_KEYS = ("problem", "prompt", "question", "instruction")
RESPONSE_KEYS = ("generated_solution", "response", "solution", "answer", "output")
ROLE_ALIASES = {
    "human": "user",
    "prompter": "user",
    "user": "user",
    "gpt": "assistant",
    "bot": "assistant",
    "model": "assistant",
    "assistant": "assistant",
    "system": "system",
}


def _first_string(record: dict, keys: Iterable[str]) -> str | None:
    for key in keys:
        value = record.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return None


def _message_text(messages) -> str | None:
    if not isinstance(messages, list):
        return None
    parts = []
    for message in messages:
        if isinstance(message, dict):
            content = message.get("content") or message.get("value")
            role = message.get("role") or message.get("from") or "unknown"
            if isinstance(content, str) and content.strip():
                parts.append(f"<|{role}|>\n{content.strip()}")
        elif isinstance(message, str) and message.strip():
            parts.append(message.strip())
    return "\n".join(parts) if parts else None


def chat_messages(record: dict) -> list[dict[str, str]] | None:
    messages: list[dict[str, str]] = []
    system = record.get("system")
    if isinstance(system, str) and system.strip():
        messages.append({"role": "system", "content": system.strip()})

    turns = record.get("messages")
    if not isinstance(turns, list):
        turns = record.get("conversations")

    if isinstance(turns, list):
        for turn in turns:
            if not isinstance(turn, dict):
                continue
            content = turn.get("content") or turn.get("value")
            if not isinstance(content, str) or not content.strip():
                continue
            raw_role = str(turn.get("role") or turn.get("from") or "user").strip().lower()
            messages.append({"role": ROLE_ALIASES.get(raw_role, "user"), "content": content.strip()})
    else:
        prompt = _first_string(record, PROMPT_KEYS)
        if not prompt:
            return None
        messages.append({"role": "user", "content": prompt})
        response = _first_string(record, RESPONSE_KEYS)
        if response:
            messages.append({"role": "assistant", "content": response})

    return messages if any(m["role"] != "system" for m in messages) else None


def chat_text(record: dict, tokenizer) -> str | None:
    """Template only the prompt and append the response verbatim.

    Templating the response would let Qwen3's chat template replace OpenThoughts'
    ``<|begin_of_thought|>`` reasoning with an empty ``<think>`` block.
    """

    messages = chat_messages(record)
    if not messages:
        return None
    split = len(messages)
    for index in range(len(messages) - 1, -1, -1):
        if messages[index]["role"] == "assistant":
            split = index
            break
    prefix = messages[:split]
    if not prefix:
        return None
    prompt = tokenizer.apply_chat_template(prefix, tokenize=False, add_generation_prompt=True)
    body = "\n".join(message["content"] for message in messages[split:])
    return f"{prompt}{body}" if body else prompt


def extract_text(record: dict, tokenizer=None, use_chat_template: bool = False) -> str | None:
    value = _first_string(record, ("text", "content", "document"))
    if value:
        return value
    if use_chat_template:
        return chat_text(record, tokenizer)
    for key in ("messages", "conversations"):
        value = _message_text(record.get(key))
        if value:
            return value
    prompt = _first_string(record, PROMPT_KEYS)
    response = _first_string(record, RESPONSE_KEYS)
    if prompt and response:
        return f"{prompt}\n\n{response}"
    return prompt


def _local_data_files(path: Path) -> tuple[str, list[str]] | None:
    """Map a local dataset dump onto a HuggingFace builder."""

    if not path.exists():
        return None
    if path.is_file():
        suffix = path.suffix.lower()
        if suffix == ".parquet":
            return "parquet", [str(path)]
        if suffix in {".arrow", ".jsonl", ".json"}:
            return ("arrow" if suffix == ".arrow" else "json"), [str(path)]
        return None
    parquet = sorted(str(file) for file in path.glob("*.parquet"))
    if not parquet:
        parquet = sorted(str(file) for file in path.glob("data/*.parquet"))
    if parquet:
        return "parquet", parquet
    arrows = sorted(
        str(file)
        for file in path.glob("*.arrow")
        if not file.name.startswith("cache-")
        and "red_pajama-data-1_t-sample-train-" in file.name
    )
    if not arrows:
        arrows = sorted(
            str(file)
            for file in path.glob("*.arrow")
            if not file.name.startswith("cache-") and not file.name.startswith("tmp")
        )
    if arrows:
        return "arrow", arrows
    for candidate in path.glob("plain_text/*/*"):
        detected = _local_data_files(candidate)
        if detected:
            return detected
    return None


def load_text_dataset(dataset_name: str, split: str, cache_dir: str | None):
    local = _local_data_files(Path(dataset_name))
    if local is not None:
        builder, files = local
        return load_dataset(builder, data_files={split: files}, split=split, streaming=True, cache_dir=cache_dir)
    return load_dataset(dataset_name, split=split, streaming=True, cache_dir=cache_dir)


def dataset_texts(
    dataset_name: str,
    split: str,
    seed: int,
    shuffle_buffer: int,
    cache_dir: str | None,
    tokenizer=None,
    use_chat_template: bool = False,
) -> Iterator[str]:
    dataset = load_text_dataset(dataset_name, split, cache_dir)
    dataset = dataset.shuffle(seed=seed, buffer_size=shuffle_buffer)
    for record in dataset:
        text = extract_text(record, tokenizer, use_chat_template)
        if text:
            yield text


def packed_sequences(
    texts: Iterable[str],
    tokenizer,
    sequence_length: int,
    max_document_tokens: int,
) -> Iterator[torch.Tensor]:
    buffer: list[int] = []
    eos_token_id = tokenizer.eos_token_id
    if eos_token_id is None:
        raise ValueError("tokenizer must define eos_token_id")
    for text in texts:
        token_ids = tokenizer.encode(text, add_special_tokens=False)
        if max_document_tokens > 0:
            token_ids = token_ids[:max_document_tokens]
        if not token_ids:
            continue
        buffer.extend(token_ids)
        buffer.append(eos_token_id)
        while len(buffer) >= sequence_length:
            yield torch.tensor(buffer[:sequence_length], dtype=torch.int32)
            del buffer[:sequence_length]


def storage_preflight(output_dir: Path, total_tokens: int) -> dict:
    existing_parent = output_dir
    while not existing_parent.exists():
        existing_parent = existing_parent.parent
    usage = shutil.disk_usage(existing_parent)
    estimated_bytes = total_tokens * torch.tensor([], dtype=torch.int32).element_size()
    return {
        "target": str(output_dir),
        "estimated_bytes": estimated_bytes,
        "free_bytes": usage.free,
        "enough_space": usage.free >= int(estimated_bytes * 1.1),
    }


def build_token_stream(args) -> None:
    output_dir = Path(args.output_dir)
    total_tokens = args.reasoning_tokens + args.general_tokens
    if args.reasoning_tokens % args.sequence_length or args.general_tokens % args.sequence_length:
        raise ValueError("per-source token targets must be divisible by sequence_length")
    preflight = storage_preflight(output_dir, total_tokens)
    print(json.dumps(preflight, indent=2))
    if args.dry_run:
        return
    if not preflight["enough_space"] and not args.allow_low_space:
        raise RuntimeError("insufficient free space; pass --allow-low-space to override")
    manifest_path = output_dir / "manifest.json"
    if manifest_path.exists() and not args.overwrite:
        manifest = json.loads(manifest_path.read_text())
        if manifest.get("complete"):
            print(f"Complete token stream already exists at {output_dir}")
            return
        raise RuntimeError("partial token stream exists; pass --overwrite to rebuild")
    if output_dir.exists() and args.overwrite:
        for path in output_dir.glob("tokens_*.pt*"):
            path.unlink()
    output_dir.mkdir(parents=True, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, trust_remote_code=args.trust_remote_code)
    chat_template = getattr(tokenizer, "chat_template", None)
    if (args.reasoning_chat_template or args.general_chat_template) and not chat_template:
        raise ValueError(
            f"{args.tokenizer} defines no chat template; "
            "pass --no-reasoning-chat-template to fall back to raw role markers"
        )
    streams = {
        "reasoning": packed_sequences(
            dataset_texts(
                args.reasoning_dataset,
                args.reasoning_split,
                args.seed,
                args.shuffle_buffer,
                args.dataset_cache_dir,
                tokenizer,
                args.reasoning_chat_template,
            ),
            tokenizer,
            args.sequence_length,
            args.max_document_tokens,
        ),
        "general": packed_sequences(
            dataset_texts(
                args.general_dataset,
                args.general_split,
                args.seed + 1,
                args.shuffle_buffer,
                args.dataset_cache_dir,
                tokenizer,
                args.general_chat_template,
            ),
            tokenizer,
            args.sequence_length,
            args.max_document_tokens,
        ),
    }
    remaining = {
        "reasoning": args.reasoning_tokens // args.sequence_length,
        "general": args.general_tokens // args.sequence_length,
    }
    rng = random.Random(args.seed)
    shard_rows = max(1, args.shard_tokens // args.sequence_length)
    global_hash = hashlib.sha256()
    shards = []
    shard_idx = 0
    rows: list[torch.Tensor] = []
    source_rows = {"reasoning": 0, "general": 0}

    def flush() -> None:
        nonlocal shard_idx, rows, source_rows
        if not rows:
            return
        tensor = torch.stack(rows)
        raw = tensor.numpy().tobytes()
        temporary = output_dir / f"tokens_{shard_idx:05d}.pt.partial"
        final = output_dir / f"tokens_{shard_idx:05d}.pt"
        torch.save({"input_ids": tensor}, temporary)
        os.replace(temporary, final)
        global_hash.update(raw)
        shards.append(
            {
                "index": shard_idx,
                "path": final.name,
                "shape": list(tensor.shape),
                "sha256": hashlib.sha256(raw).hexdigest(),
                "source_rows": dict(source_rows),
            }
        )
        shard_idx += 1
        rows = []
        source_rows = {"reasoning": 0, "general": 0}
        _atomic_json(
            manifest_path,
            {
                "complete": False,
                "sequence_length": args.sequence_length,
                "total_tokens": sum(shard["shape"][0] * shard["shape"][1] for shard in shards),
                "shards": shards,
            },
        )

    def pull(source: str) -> None:
        try:
            rows.append(next(streams[source]))
        except StopIteration as error:
            raise RuntimeError(f"{source} dataset ended before reaching its token target") from error
        source_rows[source] += 1
        remaining[source] -= 1
        if len(rows) >= shard_rows:
            flush()

    if args.source_order == "sequential":
        for source in ("reasoning", "general"):
            while remaining[source]:
                pull(source)
    else:
        while remaining["reasoning"] or remaining["general"]:
            available = [name for name, count in remaining.items() if count > 0]
            pull(rng.choice(available))
    flush()
    _atomic_json(
        manifest_path,
        {
            "complete": True,
            "tokenizer": args.tokenizer,
            "tokenizer_vocab_size": len(tokenizer),
            "tokenizer_vocab_sha256": vocab_fingerprint(tokenizer),
            "reasoning_dataset": args.reasoning_dataset,
            "general_dataset": args.general_dataset,
            "reasoning_tokens": args.reasoning_tokens,
            "general_tokens": args.general_tokens,
            "sequence_length": args.sequence_length,
            "total_tokens": total_tokens,
            "seed": args.seed,
            "source_order": args.source_order,
            "reasoning_chat_template": args.reasoning_chat_template,
            "general_chat_template": args.general_chat_template,
            "chat_template_sha256": hashlib.sha256(chat_template.encode()).hexdigest() if chat_template else None,
            "token_stream_hash": global_hash.hexdigest(),
            "shards": shards,
        },
    )
    print(f"Wrote {total_tokens:,} aligned tokens to {output_dir}")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--tokenizer", required=True, help="any model of the tokenizer family")
    parser.add_argument("--reasoning-dataset", default=DEFAULT_REASONING)
    parser.add_argument("--reasoning-split", default="train")
    parser.add_argument("--reasoning-tokens", type=int, default=200_000_000)
    parser.add_argument("--general-dataset", default=DEFAULT_GENERAL)
    parser.add_argument("--general-split", default="train")
    parser.add_argument("--general-tokens", type=int, default=200_000_000)
    parser.add_argument("--reasoning-chat-template", dest="reasoning_chat_template", action="store_true", default=True)
    parser.add_argument("--no-reasoning-chat-template", dest="reasoning_chat_template", action="store_false")
    parser.add_argument("--general-chat-template", dest="general_chat_template", action="store_true", default=False)
    parser.add_argument(
        "--source-order",
        choices=("sequential", "interleaved"),
        default="sequential",
        help="sequential writes all reasoning rows before the general rows, so a shard's source follows from its index",
    )
    parser.add_argument("--sequence-length", type=int, default=512)
    parser.add_argument("--shard-tokens", type=int, default=1_000_000)
    parser.add_argument("--max-document-tokens", type=int, default=32768)
    parser.add_argument("--shuffle-buffer", type=int, default=10000)
    parser.add_argument("--dataset-cache-dir")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--trust-remote-code", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--allow-low-space", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    build_token_stream(parse_args())
