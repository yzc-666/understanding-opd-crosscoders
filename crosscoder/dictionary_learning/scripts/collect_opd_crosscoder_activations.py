"""Cache one model's residual-stream activations on the shared token stream."""

from __future__ import annotations

import argparse
import json
import os
import shutil
from pathlib import Path

import torch
from transformers import AutoConfig, AutoModelForCausalLM

ROLES = ("base", "opd", "teacher")


def _atomic_json(path: Path, value: dict) -> None:
    temporary = path.with_suffix(f"{path.suffix}.partial")
    with temporary.open("w") as handle:
        json.dump(value, handle, indent=2, sort_keys=True)
    os.replace(temporary, path)


def _torch_dtype(name: str) -> torch.dtype:
    return {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}[name]


def _load_token_manifest(token_dir: Path) -> dict:
    manifest_path = token_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    if not manifest.get("complete"):
        raise RuntimeError(f"token stream is incomplete: {manifest_path}")
    return manifest


def _shard_is_complete(
    output_dir: Path,
    shard_idx: int,
    expected_shape: tuple[int, int],
    token_sha256: str,
    token_stream_hash: str | None,
    model_path: str,
    layer: int,
) -> bool:
    data_path = output_dir / f"shard_{shard_idx}.memmap"
    meta_path = output_dir / f"shard_{shard_idx}.meta"
    if not data_path.exists() or not meta_path.exists():
        return False
    metadata = json.loads(meta_path.read_text())
    expected_bytes = expected_shape[0] * expected_shape[1] * 2
    return (
        metadata.get("complete") is True
        and tuple(metadata.get("shape", ())) == expected_shape
        and metadata.get("token_shard_sha256") == token_sha256
        and metadata.get("token_stream_hash") == token_stream_hash
        and metadata.get("model_path") == model_path
        and metadata.get("layer") == layer
        and data_path.stat().st_size == expected_bytes
    )


def _check_tokenizer(tokenizer_path: str, manifest: dict, strict: bool) -> None:
    """Refuse a model whose vocabulary differs from the one the token stream was built with."""

    expected_sha = manifest.get("tokenizer_vocab_sha256")
    expected_size = manifest.get("tokenizer_vocab_size")
    if expected_sha is None and expected_size is None:
        print("Token manifest records no tokenizer identity; skipping vocab check")
        return
    try:
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, trust_remote_code=True)
    except Exception as error:
        message = f"cannot load a tokenizer from {tokenizer_path}: {type(error).__name__}"
        if strict:
            raise RuntimeError(message) from error
        print(f"Warning: {message}; skipping vocab check")
        return

    if expected_sha is not None:
        from dictionary_learning.scripts.prepare_opd_crosscoder_tokens import vocab_fingerprint

        actual = vocab_fingerprint(tokenizer)
        if actual != expected_sha:
            raise RuntimeError(
                f"tokenizer mismatch: {tokenizer_path} has vocab sha {actual[:12]} "
                f"but the token stream was built with {expected_sha[:12]}"
            )
    elif len(tokenizer) != expected_size:
        raise RuntimeError(
            f"tokenizer mismatch: {tokenizer_path} has {len(tokenizer)} entries "
            f"but the token stream was built with {expected_size}"
        )
    print(f"Tokenizer check passed for {tokenizer_path}")


def _storage_preflight(output_dir: Path, estimated_bytes: int) -> dict:
    parent = output_dir
    while not parent.exists():
        parent = parent.parent
    usage = shutil.disk_usage(parent)
    return {
        "target": str(output_dir),
        "estimated_bytes": estimated_bytes,
        "free_bytes": usage.free,
        "enough_space": usage.free >= int(estimated_bytes * 1.05),
    }


def _try_finalize(output_dir: Path, manifest: dict, hidden_size: int, model_path: str, layer: int, args) -> bool:
    """Write mean, std, and config.json once every shard of the stream is cached."""

    shard_stats = []
    for shard in manifest["shards"]:
        shard_idx = int(shard["index"])
        row_count, sequence_length = map(int, shard["shape"])
        token_count = int(shard.get("valid_tokens", row_count * sequence_length))
        stats_path = output_dir / f"shard_{shard_idx}.stats.pt"
        if not _shard_is_complete(
            output_dir,
            shard_idx,
            (token_count, hidden_size),
            shard["sha256"],
            manifest["token_stream_hash"],
            model_path,
            layer,
        ) or not stats_path.exists():
            return False
        shard_stats.append(torch.load(stats_path, weights_only=True, map_location="cpu"))
    count = sum(int(value["count"]) for value in shard_stats)
    total = sum((value["sum"] for value in shard_stats), torch.zeros(hidden_size))
    total_square = sum((value["sum_square"] for value in shard_stats), torch.zeros(hidden_size))
    mean = total / count
    variance = ((total_square - count * mean.square()) / max(count - 1, 1)).clamp_min(0)
    torch.save(mean.float(), output_dir / "mean.pt")
    torch.save(variance.sqrt().float(), output_dir / "std.pt")
    _atomic_json(
        output_dir / "config.json",
        {
            "batch_size": args.batch_size,
            "context_len": manifest["sequence_length"],
            "shard_size": None,
            "d_model": hidden_size,
            "shuffle_shards": False,
            "io": "out",
            "total_size": count,
            "shard_count": len(manifest["shards"]),
            "store_tokens": False,
            "store_sequence_ranges": False,
            "token_stream_hash": manifest["token_stream_hash"],
            "model_path": model_path,
            "tokenizer_check_path": getattr(args, "tokenizer", None) or model_path,
            "role": args.role,
            "layer": layer,
            "dtype": "torch.bfloat16",
            "complete": True,
            "world_size": args.world_size,
        },
    )
    print(f"Cached {count:,} activations for {args.role} in {output_dir}")
    return True


def collect(args) -> None:
    if args.world_size < 1 or not 0 <= args.rank < args.world_size:
        raise ValueError(f"invalid rank/world-size: {args.rank}/{args.world_size}")
    model_path = os.path.abspath(args.model)
    layer = args.layer
    token_dir = Path(args.token_dir)
    manifest = _load_token_manifest(token_dir)
    config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
    hidden_size = int(config.hidden_size)
    if not 0 <= layer < int(config.num_hidden_layers):
        raise ValueError(f"layer {layer} is invalid for {config.num_hidden_layers} layers")
    output_dir = Path(args.output_dir)
    preflight = _storage_preflight(output_dir, int(manifest["total_tokens"]) * hidden_size * 2)
    preflight.update(
        {
            "role": args.role,
            "model": model_path,
            "layer": layer,
            "hidden_size": hidden_size,
            "tokens": manifest["total_tokens"],
            "rank": args.rank,
            "world_size": args.world_size,
        }
    )
    print(json.dumps(preflight, indent=2))
    if not args.skip_tokenizer_check and manifest["shards"]:
        _check_tokenizer(args.tokenizer or model_path, manifest, args.strict_tokenizer_check)
    if args.dry_run:
        return
    if not preflight["enough_space"] and not args.allow_low_space:
        raise RuntimeError("insufficient free space; pass --allow-low-space to override")
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.finalize_only:
        if not _try_finalize(output_dir, manifest, hidden_size, model_path, layer, args):
            raise RuntimeError(f"cannot finalize {args.role}: some shards are still incomplete")
        return

    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=_torch_dtype(args.dtype),
        device_map=args.device_map,
        attn_implementation=args.attn_implementation,
        trust_remote_code=True,
    )
    model.eval()
    from dictionary_learning.cache import IncrementalActivationShardWriter

    backbone = getattr(model, "model", model)
    captured: dict[str, torch.Tensor] = {}

    def capture_output(_module, _inputs, output):
        captured["activation"] = output[0] if isinstance(output, tuple) else output

    hook = backbone.layers[layer].register_forward_hook(capture_output)
    input_device = model.get_input_embeddings().weight.device
    try:
        for shard in manifest["shards"]:
            shard_idx = int(shard["index"])
            if shard_idx % args.world_size != args.rank:
                continue
            row_count, sequence_length = map(int, shard["shape"])
            token_count = int(shard.get("valid_tokens", row_count * sequence_length))
            expected_shape = (token_count, hidden_size)
            if not args.overwrite and _shard_is_complete(
                output_dir,
                shard_idx,
                expected_shape,
                shard["sha256"],
                manifest["token_stream_hash"],
                model_path,
                layer,
            ):
                print(f"Skipping complete shard {shard_idx}")
                continue
            token_payload = torch.load(token_dir / shard["path"], map_location="cpu", weights_only=True)
            input_ids = token_payload["input_ids"].to(dtype=torch.long)
            attention_mask = token_payload.get("attention_mask")
            if attention_mask is not None:
                attention_mask = attention_mask.to(dtype=torch.long)
            writer = IncrementalActivationShardWriter(
                str(output_dir),
                shard_idx,
                expected_shape,
                dtype=torch.bfloat16,
                metadata={
                    "role": args.role,
                    "model_path": model_path,
                    "layer": layer,
                    "token_shard_sha256": shard["sha256"],
                    "token_stream_hash": manifest["token_stream_hash"],
                },
            )
            count = 0
            total = torch.zeros(hidden_size, dtype=torch.float64)
            total_square = torch.zeros(hidden_size, dtype=torch.float64)
            try:
                for start in range(0, row_count, args.batch_size):
                    batch = input_ids[start : start + args.batch_size].to(input_device)
                    batch_mask = (
                        attention_mask[start : start + args.batch_size].to(input_device)
                        if attention_mask is not None
                        else None
                    )
                    captured.clear()
                    with torch.inference_mode():
                        backbone(input_ids=batch, attention_mask=batch_mask, use_cache=False, return_dict=True)
                    flat = captured.pop("activation").detach().reshape(-1, hidden_size)
                    if batch_mask is not None:
                        flat = flat[batch_mask.reshape(-1).bool()]
                    writer.write(flat)
                    float_activation = flat.float()
                    total += float_activation.sum(dim=0).cpu().double()
                    total_square += float_activation.square().sum(dim=0).cpu().double()
                    count += flat.shape[0]
                    del flat, float_activation
                writer.finalize()
            except Exception:
                writer.abort()
                raise
            torch.save(
                {"count": count, "sum": total, "sum_square": total_square},
                output_dir / f"shard_{shard_idx}.stats.pt",
            )
    finally:
        hook.remove()
        del model

    if not _try_finalize(output_dir, manifest, hidden_size, model_path, layer, args):
        print(
            f"Rank {args.rank}/{args.world_size} finished its shards for {args.role}; "
            "run with --finalize-only once all ranks are done"
        )


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--role", choices=ROLES, required=True, help="crosscoder slot this model fills")
    parser.add_argument("--model", required=True)
    parser.add_argument("--tokenizer", help="tokenizer used to validate token ids (defaults to --model)")
    parser.add_argument("--layer", type=int, required=True, help="block whose output is cached, counting from 0")
    parser.add_argument("--token-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--batch-size", type=int, default=64, help="sequences per forward pass")
    parser.add_argument("--dtype", choices=("bfloat16", "float16", "float32"), default="bfloat16")
    parser.add_argument("--device-map", default="auto")
    parser.add_argument("--attn-implementation", default="sdpa")
    parser.add_argument("--rank", type=int, default=0)
    parser.add_argument("--world-size", type=int, default=1)
    parser.add_argument("--finalize-only", action="store_true")
    parser.add_argument("--skip-tokenizer-check", action="store_true")
    parser.add_argument("--strict-tokenizer-check", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--allow-low-space", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    collect(parse_args())
