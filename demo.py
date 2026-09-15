"""Run a small CPU smoke demo, or generate text from an existing checkpoint."""

import argparse
from pathlib import Path

import torch
from tokenizers import Tokenizer

from model import ModelConfig, TinyTransformer

ROOT = Path(__file__).resolve().parent


def positive_int(value):
    value = int(value)
    if value < 1:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return value


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--checkpoint",
        type=Path,
        help="local state dictionary; default architecture matches experiment 2",
    )
    parser.add_argument(
        "--tokenizer", type=Path, default=ROOT / "tiny_shakespeare_bpe.json"
    )
    parser.add_argument("--corpus", type=Path, default=ROOT / "input.txt")
    parser.add_argument("--steps", type=positive_int, default=5)
    parser.add_argument("--tokens", type=positive_int, default=32)
    parser.add_argument("--embedding-dim", type=positive_int, default=64)
    parser.add_argument("--heads", type=positive_int, default=4)
    parser.add_argument("--layers", type=positive_int, default=4)
    parser.add_argument("--context", type=positive_int, default=64)
    parser.add_argument("--prompt", default="O Romeo, Romeo! ")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args(argv)
    torch.set_num_threads(1)
    torch.manual_seed(args.seed)
    try:
        tokenizer = Tokenizer.from_file(str(args.tokenizer))
        config = ModelConfig(
            vocab_size=tokenizer.get_vocab_size(),
            block_size=args.context,
            n_embd=args.embedding_dim,
            n_head=args.heads,
            n_layer=args.layers,
        )
        model = TinyTransformer(config)
        prompt = tokenizer.encode(args.prompt).ids
        if not prompt:
            raise ValueError("prompt must encode to at least one token")
        if args.checkpoint:
            model.load_state_dict(
                torch.load(args.checkpoint, map_location="cpu", weights_only=True)
            )
            print(f"Loaded checkpoint: {args.checkpoint.name}")
        else:
            # A tiny functional training check, not a validation benchmark.
            with args.corpus.open(encoding="utf-8") as handle:
                corpus = handle.read(10_000)
            data = torch.tensor(tokenizer.encode(corpus).ids, dtype=torch.long)
            if len(data) <= config.block_size:
                raise ValueError(
                    "corpus must encode to more tokens than the context length"
                )
            optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
            print(
                f"CPU smoke check: {args.steps} training steps; generated text is not a quality evaluation."
            )
            for step in range(args.steps):
                starts = torch.randint(len(data) - config.block_size, (4,))
                x = torch.stack([data[i : i + config.block_size] for i in starts])
                y = torch.stack(
                    [data[i + 1 : i + config.block_size + 1] for i in starts]
                )
                _, loss = model(x, y)
                if not torch.isfinite(loss):
                    raise ValueError("non-finite training loss")
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                if any(
                    p.grad is not None and not torch.isfinite(p.grad).all()
                    for p in model.parameters()
                ):
                    raise ValueError("non-finite gradient")
                optimizer.step()
                print(f"step {step + 1}: loss={loss.item():.4f}")
        generated = model.generate(
            torch.tensor([prompt], dtype=torch.long), args.tokens
        )
        print(
            f"Parameters: {sum(p.numel() for p in model.parameters()):,}; device: CPU"
        )
        print("Generated sample:")
        print(tokenizer.decode(generated[0].tolist()))
    except (OSError, ValueError, RuntimeError) as exc:
        parser.error(str(exc))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
