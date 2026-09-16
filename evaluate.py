"""2026-09-16 AI-assisted maintenance: train-only BPE and held-out evaluation.

This is a new reproducibility workflow, not a reproduction of the original
coursework scores. Raw UTF-8 text is split sequentially before tokenizer fitting.
Every run evaluates its fixed final training step; there is no model selection.
"""

import argparse
import hashlib
import importlib.metadata
import json
import math
import os
import platform
import statistics
import sys
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path

import torch
from tokenizers import Tokenizer, decoders, models, pre_tokenizers, trainers
from torch.nn import functional as F

from model import ModelConfig, TinyTransformer

ROOT = Path(__file__).resolve().parent
SPLIT_NAMES = ("train", "validation", "test")


def positive_integer(value, name):
    if type(value) is not int or value < 1:
        raise ValueError(f"{name} must be a positive integer")
    return value


def sha256(data):
    return hashlib.sha256(data).hexdigest()


def canonical_json(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def split_raw_text(text):
    """Split Unicode characters at floor(0.8*N) and floor(0.9*N).

    UTF-8 is decoded strictly by the CLI, with no newline normalization. Splits
    may cut words/documents but never a Unicode code point or encoded byte.
    """
    if not isinstance(text, str):
        raise TypeError("corpus must be text")
    text.encode("utf-8", errors="strict")
    first, second = len(text) * 8 // 10, len(text) * 9 // 10
    parts = (text[:first], text[first:second], text[second:])
    if any(not part for part in parts):
        raise ValueError("corpus is too short for three nonempty raw-text splits")
    return dict(zip(SPLIT_NAMES, parts))


def fit_tokenizer(train_text, vocab_size=500):
    """Fit BPE on training text only; all 256 byte symbols are unconditional.

    No unknown/special tokens, normalization, byte-prefix insertion or dropout
    are used. The full byte alphabet can represent unseen valid UTF-8 text.
    """
    positive_integer(vocab_size, "vocab_size")
    if vocab_size < 256:
        raise ValueError("vocab_size must be at least 256 for the full byte alphabet")
    if not isinstance(train_text, str) or not train_text:
        raise ValueError("tokenizer training text must be nonempty")
    train_text.encode("utf-8", errors="strict")
    tokenizer = Tokenizer(models.BPE())
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tokenizer.decoder = decoders.ByteLevel()
    trainer = trainers.BpeTrainer(
        vocab_size=vocab_size,
        min_frequency=2,
        initial_alphabet=sorted(pre_tokenizers.ByteLevel.alphabet()),
        special_tokens=[],
        show_progress=False,
    )
    tokenizer.train_from_iterator([train_text], trainer=trainer)
    return tokenizer


@dataclass
class PreparedCorpus:
    tokenizer: Tokenizer
    tokenizer_json: str
    ids: dict
    metadata: dict


def prepare_corpus(text, vocab_size=500):
    parts = split_raw_text(text)
    tokenizer = fit_tokenizer(parts["train"], vocab_size)
    serialized = canonical_json(json.loads(tokenizer.to_str()))
    ids, metadata = {}, {}
    offset = 0
    for name, part in parts.items():
        encoded = tokenizer.encode(part, add_special_tokens=False).ids
        if len(encoded) < 2:
            raise ValueError(
                f"{name} split needs at least two tokens for next-token loss"
            )
        if tokenizer.decode(encoded) != part:
            raise ValueError(f"tokenizer does not round-trip the {name} partition")
        ids[name] = torch.tensor(encoded, dtype=torch.long)
        metadata[name] = {
            "character_start_inclusive": offset,
            "character_end_exclusive": offset + len(part),
            "characters": len(part),
            "utf8_bytes": len(part.encode("utf-8")),
            "text_sha256": sha256(part.encode("utf-8")),
            "token_ids_sha256": sha256(canonical_json(encoded).encode("utf-8")),
            "tokens": len(encoded),
            "next_token_targets": len(encoded) - 1,
        }
        offset += len(part)
    return PreparedCorpus(tokenizer, serialized, ids, metadata)


def validate_ids(ids, minimum=2):
    if not isinstance(ids, torch.Tensor) or ids.ndim != 1 or ids.dtype != torch.long:
        raise ValueError("token IDs must be a one-dimensional torch.long tensor")
    if ids.device.type != "cpu":
        raise ValueError("stored corpus token IDs must remain on CPU")
    if len(ids) < minimum or bool((ids < 0).any()):
        raise ValueError(f"need at least {minimum} nonnegative token IDs")


def sample_training_batch(ids, context, batch_size, generator, device="cpu"):
    """Random contexts and their targets remain wholly inside this train split."""
    positive_integer(context, "context")
    positive_integer(batch_size, "batch_size")
    validate_ids(ids, context + 1)
    if not isinstance(generator, torch.Generator) or generator.device.type != "cpu":
        raise ValueError("training-data generator must be an explicit CPU generator")
    starts = torch.randint(len(ids) - context, (batch_size,), generator=generator)
    x = torch.stack([ids[int(i) : int(i) + context] for i in starts])
    y = torch.stack([ids[int(i) + 1 : int(i) + context + 1] for i in starts])
    return x.to(device), y.to(device)


def evaluation_windows(ids, context):
    """Fixed windows cover target indices 1..N-1 once, including the final tail.

    Input positions have stride=context and do not overlap. Each window resets
    positional/context history; one boundary token is a target in the preceding
    window and an input in the following window, never a duplicate target.
    """
    positive_integer(context, "context")
    validate_ids(ids)
    for start in range(0, len(ids) - 1, context):
        stop = min(start + context, len(ids) - 1)
        yield ids[start:stop], ids[start + 1 : stop + 1]


def evaluation_batches(ids, context, batch_size):
    positive_integer(batch_size, "evaluation batch_size")
    batch = []
    for x, y in evaluation_windows(ids, context):
        if batch and (len(batch) == batch_size or len(x) != len(batch[0][0])):
            yield torch.stack([a for a, _ in batch]), torch.stack([b for _, b in batch])
            batch = []
        batch.append((x, y))
    if batch:
        yield torch.stack([a for a, _ in batch]), torch.stack([b for _, b in batch])


def evaluate_model(model, ids, batch_size=16, device="cpu"):
    """Token-weighted NLL; preserve training flags and CPU/CUDA dropout RNG.

    Evaluation consumes no training-data generator. RNG restoration also holds
    if a future evaluation-only model component happens to draw random numbers.
    """
    device = torch.device(device)
    validate_ids(ids)
    if int(ids.max()) >= model.config.vocab_size:
        raise ValueError("evaluation token ID is outside the model vocabulary")
    flags = [(module, module.training) for module in model.modules()]
    devices = [device.index or 0] if device.type == "cuda" else []
    total_nll, total_tokens, windows = 0.0, 0, 0
    try:
        with torch.random.fork_rng(devices=devices), torch.inference_mode():
            model.eval()
            for x, y in evaluation_batches(ids, model.config.block_size, batch_size):
                logits, _ = model(x.to(device))
                token_losses = F.cross_entropy(
                    logits.reshape(-1, model.config.vocab_size).float(),
                    y.to(device).reshape(-1),
                    reduction="none",
                )
                if not bool(torch.isfinite(token_losses).all()):
                    raise ValueError("non-finite held-out loss")
                total_nll += token_losses.double().sum().item()
                total_tokens += y.numel()
                windows += len(x)
    finally:
        for module, was_training in flags:
            module.training = was_training
    loss = total_nll / total_tokens
    return {
        "nll_sum_nats": total_nll,
        "next_token_targets": total_tokens,
        "windows": windows,
        "loss_nats_per_token": loss,
        "perplexity": math.exp(loss) if loss < 709 else None,
    }


@dataclass(frozen=True)
class RunConfig:
    steps: int = 100
    batch_size: int = 8
    learning_rate: float = 0.001
    weight_decay: float = 0.01
    gradient_clip: float = 1.0
    evaluation_batch_size: int = 16
    device: str = "cpu"

    def __post_init__(self):
        for name in ("steps", "batch_size", "evaluation_batch_size"):
            positive_integer(getattr(self, name), name)
        for name in ("learning_rate", "gradient_clip"):
            value = getattr(self, name)
            if not math.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be finite and positive")
        if not math.isfinite(self.weight_decay) or self.weight_decay < 0:
            raise ValueError("weight_decay must be finite and nonnegative")
        if self.device not in ("cpu", "cuda"):
            raise ValueError("device must be cpu or cuda")


def model_state_hash(model):
    """Hash tensor content, not nondeterministic archive serialization metadata."""
    digest = hashlib.sha256()
    for name, tensor in sorted(model.state_dict().items()):
        tensor = tensor.detach().cpu().contiguous()
        digest.update(
            canonical_json([name, str(tensor.dtype), list(tensor.shape)]).encode()
        )
        digest.update(bytes(tensor.view(torch.uint8).reshape(-1).tolist()))
    return digest.hexdigest()


def run_seed(prepared, model_config, run_config, seed):
    if type(seed) is not int or not 0 <= seed < 2**63:
        raise ValueError("seeds must be integers in [0, 2**63)")
    if model_config.vocab_size != prepared.tokenizer.get_vocab_size():
        raise ValueError("model and tokenizer vocabularies must match")
    validate_ids(prepared.ids["train"], model_config.block_size + 1)
    devices = [0] if run_config.device == "cuda" else []
    with torch.random.fork_rng(devices=devices):
        torch.manual_seed(seed)
        data_generator = torch.Generator(device="cpu").manual_seed(seed)
        model = TinyTransformer(model_config).to(run_config.device).train()
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=run_config.learning_rate,
            weight_decay=run_config.weight_decay,
            foreach=False,
            fused=False,
        )
        losses = []
        for _ in range(run_config.steps):
            x, y = sample_training_batch(
                prepared.ids["train"],
                model_config.block_size,
                run_config.batch_size,
                data_generator,
                run_config.device,
            )
            optimizer.zero_grad(set_to_none=True)
            _, loss = model(x, y)
            if not bool(torch.isfinite(loss)):
                raise ValueError("non-finite training loss")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                run_config.gradient_clip,
                error_if_nonfinite=True,
            )
            optimizer.step()
            losses.append(loss.item())
        final_state_sha256 = model_state_hash(model)
        return {
            "seed": seed,
            "training_step_losses_nats_per_token": losses,
            "model_state_sha256": final_state_sha256,
            "parameters": sum(p.numel() for p in model.parameters()),
            "validation": evaluate_model(
                model,
                prepared.ids["validation"],
                run_config.evaluation_batch_size,
                run_config.device,
            ),
            "test": evaluate_model(
                model,
                prepared.ids["test"],
                run_config.evaluation_batch_size,
                run_config.device,
            ),
        }


def summarize(results):
    summary = {}
    for split in ("validation", "test"):
        summary[split] = {}
        for metric in ("loss_nats_per_token", "perplexity"):
            values = [row[split][metric] for row in results]
            usable = bool(values) and all(value is not None for value in values)
            summary[split][metric] = {
                "mean": statistics.mean(values) if usable else None,
                "sample_standard_deviation": statistics.stdev(values)
                if usable and len(values) > 1
                else None,
            }
    return summary


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus", type=Path, default=ROOT / "input.txt")
    parser.add_argument(
        "--output-dir", type=Path, required=True, help="must not already exist"
    )
    parser.add_argument("--seeds", type=int, nargs="+", default=[17, 42, 2026])
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--evaluation-batch-size", type=int, default=16)
    parser.add_argument("--learning-rate", type=float, default=0.001)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--gradient-clip", type=float, default=1.0)
    parser.add_argument("--vocab-size", type=int, default=500)
    parser.add_argument("--embedding-dim", type=int, default=64)
    parser.add_argument("--heads", type=int, default=4)
    parser.add_argument("--layers", type=int, default=2)
    parser.add_argument("--context", type=int, default=64)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    args = parser.parse_args(argv)
    try:
        if args.output_dir.exists():
            raise ValueError("output directory already exists; use a new directory")
        positive_integer(args.threads, "threads")
        if len(set(args.seeds)) != len(args.seeds) or any(
            not 0 <= s < 2**63 for s in args.seeds
        ):
            raise ValueError("use distinct seeds in [0, 2**63)")
        run_config = RunConfig(
            args.steps,
            args.batch_size,
            args.learning_rate,
            args.weight_decay,
            args.gradient_clip,
            args.evaluation_batch_size,
            args.device,
        )
        torch.set_num_threads(args.threads)
        os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
        if args.device == "cuda":
            os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
            if os.environ["CUBLAS_WORKSPACE_CONFIG"] not in (":4096:8", ":16:8"):
                raise ValueError("CUDA requires deterministic CUBLAS_WORKSPACE_CONFIG")
            if not torch.cuda.is_available():
                raise ValueError("CUDA requested but unavailable")
            torch.backends.cuda.matmul.allow_tf32 = False
            torch.backends.cudnn.allow_tf32 = False
            torch.backends.cudnn.benchmark = False
            torch.backends.cudnn.deterministic = True
        torch.use_deterministic_algorithms(True)
        raw = args.corpus.read_bytes()
        text = raw.decode("utf-8", errors="strict")
        prepared = prepare_corpus(text, args.vocab_size)
        model_config = ModelConfig(
            vocab_size=prepared.tokenizer.get_vocab_size(),
            block_size=args.context,
            n_embd=args.embedding_dim,
            n_head=args.heads,
            n_layer=args.layers,
            dropout=args.dropout,
        )
        validate_ids(prepared.ids["train"], model_config.block_size + 1)
        metadata = {
            "schema_version": 1,
            "maintenance_date": "2026-09-16",
            "attribution": "New AI-assisted maintenance workflow, not original course results.",
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "corpus": {
                "filename": args.corpus.name,
                "utf8_bytes": len(raw),
                "characters": len(text),
                "sha256": sha256(raw),
            },
            "split_method": "Sequential raw Unicode characters: [0,floor(.8*N)), [floor(.8*N),floor(.9*N)), [floor(.9*N),N); no newline normalization.",
            "splits": prepared.metadata,
            "tokenizer": {
                "fit_split": "train",
                "kind": "byte-level BPE",
                "requested_vocab_size": args.vocab_size,
                "actual_vocab_size": prepared.tokenizer.get_vocab_size(),
                "complete_byte_alphabet": True,
                "min_frequency": 2,
                "sha256": sha256(prepared.tokenizer_json.encode("utf-8")),
            },
            "model_config": asdict(model_config),
            "training_config": asdict(run_config),
            "seeds": args.seeds,
            "selection": "Fixed final step for every seed; validation/test are never used for selection or optimization.",
            "evaluation": "Each split's next-token targets1..N-1 scored once; fixed context-stride windows plus tail, reset window context, token-weighted natural-log loss. Perplexity=exp(loss).",
            "limitations": [
                "One sequential split of one corpus is not a broad language-model benchmark.",
                "Text/document overlap or repeated content across raw partitions is not deduplicated.",
                "Reproducibility is scoped to identical code, corpus, tokenizer, configuration and software/hardware; no cross-device/version bitwise guarantee.",
                "Repeated seeds estimate optimization variability, not corpus/split uncertainty; sample standard deviation is undefined for one seed.",
                "No weights, historical checkpoint, network download or architecture-selection procedure is used.",
            ],
            "source_sha256": {
                name: sha256((ROOT / name).read_bytes())
                for name in (
                    "evaluate.py",
                    "model.py",
                    "requirements.txt",
                    "requirements-evaluation.txt",
                )
            },
            "environment": {
                "python": sys.version,
                "platform": platform.platform(),
                "machine": platform.machine(),
                "torch": torch.__version__,
                "tokenizers": importlib.metadata.version("tokenizers"),
                "device": args.device,
                "device_name": torch.cuda.get_device_name(0)
                if args.device == "cuda"
                else platform.processor(),
                "torch_threads": torch.get_num_threads(),
                "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
                "cuda_runtime": torch.version.cuda if args.device == "cuda" else None,
                "cublas_workspace_config": os.environ.get("CUBLAS_WORKSPACE_CONFIG")
                if args.device == "cuda"
                else None,
            },
        }
        args.output_dir.mkdir(parents=True, exist_ok=False)
        with (args.output_dir / "tokenizer.json").open("x", encoding="utf-8") as handle:
            handle.write(prepared.tokenizer_json)
        with (args.output_dir / "metadata.json").open("x", encoding="utf-8") as handle:
            json.dump(metadata, handle, indent=2, allow_nan=False)
            handle.write("\n")
        results = []
        with (args.output_dir / "seed-results.jsonl").open(
            "x", encoding="utf-8"
        ) as handle:
            for seed in args.seeds:
                result = run_seed(prepared, model_config, run_config, seed)
                results.append(result)
                handle.write(json.dumps(result, sort_keys=True, allow_nan=False) + "\n")
                handle.flush()
                print(
                    f"seed={seed} validation={result['validation']['loss_nats_per_token']:.6f} test={result['test']['loss_nats_per_token']:.6f}",
                    flush=True,
                )
        report = {**metadata, "results": results, "summary": summarize(results)}
        with (args.output_dir / "results.json").open("x", encoding="utf-8") as handle:
            json.dump(report, handle, indent=2, allow_nan=False)
            handle.write("\n")
    except (OSError, ValueError, RuntimeError, UnicodeError) as exc:
        parser.error(str(exc))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
