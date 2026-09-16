"""Tests for the 2026-09-16 AI-assisted held-out evaluation workflow."""

import contextlib
import io
import json
import math
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import torch
from torch import nn

from evaluate import (
    RunConfig,
    canonical_json,
    evaluate_model,
    evaluation_batches,
    evaluation_windows,
    fit_tokenizer,
    main,
    prepare_corpus,
    run_seed,
    sample_training_batch,
    sha256,
    split_raw_text,
    summarize,
)
from model import ModelConfig, TinyTransformer

ROOT = Path(__file__).resolve().parents[1]
torch.set_num_threads(1)
TEXT = "the little cat and the little dog share a warm home.\n" * 32


class ConstantModel(nn.Module):
    """Known logits permit a reference loss calculation without torch CE."""

    def __init__(self, draw_random=False, fail=False):
        super().__init__()
        self.config = SimpleNamespace(vocab_size=7, block_size=4)
        self.register_buffer("scores", torch.tensor([5.0, 0, 0, 0, 0, 0, 0]))
        self.dropout = nn.Dropout(0.5)
        self.draw_random = draw_random
        self.fail = fail

    def forward(self, ids):
        if self.draw_random:
            torch.rand(13)
        if self.fail:
            raise RuntimeError("deliberate test failure")
        return self.scores.expand(*ids.shape, -1), None


class EvaluationTests(unittest.TestCase):
    def test_raw_split_rejoins_exactly_and_uses_character_boundaries(self):
        text = ("aé漢🦉\r\n\x00" * 7) + "end"
        parts = split_raw_text(text)
        self.assertEqual("".join(parts.values()), text)
        self.assertEqual(parts["train"], text[: len(text) * 8 // 10])
        self.assertEqual(
            parts["validation"], text[len(text) * 8 // 10 : len(text) * 9 // 10]
        )
        self.assertEqual(parts["test"], text[len(text) * 9 // 10 :])
        for part in parts.values():
            self.assertEqual(part.encode("utf-8").decode("utf-8"), part)

    def test_heldout_changes_cannot_change_tokenizer_or_train_ids(self):
        prefix = "alpha beta gamma delta. " * 80
        self.assertEqual(len(prefix) % 8, 0)
        heldout_chars = len(prefix) // 4
        first = prepare_corpus(prefix + "🦉" * heldout_chars, 300)
        second = prepare_corpus(prefix + "Z" * heldout_chars, 300)
        self.assertEqual(first.metadata["train"], second.metadata["train"])
        self.assertEqual(first.tokenizer_json, second.tokenizer_json)
        self.assertTrue(torch.equal(first.ids["train"], second.ids["train"]))
        self.assertNotEqual(
            first.metadata["test"]["text_sha256"],
            second.metadata["test"]["text_sha256"],
        )

    def test_each_partition_is_encoded_separately(self):
        prepared = prepare_corpus(TEXT, 300)
        offset = 0
        for name, raw in split_raw_text(TEXT).items():
            expected = prepared.tokenizer.encode(raw, add_special_tokens=False).ids
            self.assertEqual(prepared.ids[name].tolist(), expected)
            self.assertEqual(
                prepared.metadata[name]["character_start_inclusive"], offset
            )
            self.assertEqual(
                prepared.metadata[name]["text_sha256"], sha256(raw.encode())
            )
            offset += len(raw)
            self.assertEqual(prepared.metadata[name]["character_end_exclusive"], offset)

    def test_complete_byte_alphabet_round_trips_unseen_unicode(self):
        tokenizer = fit_tokenizer("aaaa bbbb aaaa bbbb " * 30, 280)
        novel = "".join(map(chr, range(256))) + "漢字🦉🚀e\u0301—\r\n\t\x00"
        encoded = tokenizer.encode(novel, add_special_tokens=False)
        self.assertTrue(encoded.ids)
        self.assertEqual(tokenizer.decode(encoded.ids), novel)
        self.assertNotIn("[UNK]", tokenizer.get_vocab())
        self.assertEqual(
            tokenizer.decode(tokenizer.encode(" no prefix").ids), " no prefix"
        )

    def test_tokenizer_is_identical_in_fresh_processes(self):
        code = (
            "import json; from evaluate import fit_tokenizer,canonical_json,sha256; "
            f"t=fit_tokenizer({TEXT!r},300); "
            "print(sha256(canonical_json(json.loads(t.to_str())).encode()))"
        )
        expected = sha256(
            canonical_json(json.loads(fit_tokenizer(TEXT, 300).to_str())).encode()
        )
        for _ in range(2):
            result = subprocess.run(
                [sys.executable, "-c", code],
                cwd=ROOT,
                check=True,
                capture_output=True,
                text=True,
                timeout=45,
            )
            self.assertEqual(result.stdout.strip(), expected)

    def test_training_batches_stay_inside_train_and_include_last_start(self):
        ids = torch.arange(80, dtype=torch.long)
        generator = torch.Generator().manual_seed(13)
        x, y = sample_training_batch(ids, 8, 4096, generator)
        self.assertEqual(tuple(x.shape), (4096, 8))
        self.assertTrue(torch.equal(y, x + 1))
        self.assertEqual(x[:, 0].min().item(), 0)
        self.assertEqual(x[:, 0].max().item(), 71)
        self.assertEqual(y.max().item(), 79)

    def test_training_generator_is_independent_of_global_rng(self):
        ids = torch.arange(50, dtype=torch.long)
        first = torch.Generator().manual_seed(9)
        second = torch.Generator().manual_seed(9)
        expected = sample_training_batch(ids, 8, 4, first)
        torch.rand(1000)
        actual = sample_training_batch(ids, 8, 4, second)
        for left, right in zip(expected, actual):
            self.assertTrue(torch.equal(left, right))

    def test_evaluation_windows_cover_targets_once_with_tail(self):
        for length in (2, 4, 5, 8, 9, 11):
            with self.subTest(length=length):
                ids = torch.arange(length, dtype=torch.long)
                windows = list(evaluation_windows(ids, 4))
                self.assertTrue(
                    torch.equal(torch.cat([y for _, y in windows]), ids[1:])
                )
                self.assertTrue(
                    torch.equal(torch.cat([x for x, _ in windows]), ids[:-1])
                )
                self.assertTrue(all(1 <= len(x) <= 4 for x, _ in windows))
                batches = list(evaluation_batches(ids, 4, 2))
                self.assertTrue(
                    torch.equal(torch.cat([y.reshape(-1) for _, y in batches]), ids[1:])
                )
        self.assertEqual(
            [len(x) for x, _ in evaluation_windows(torch.arange(11), 4)], [4, 4, 2]
        )

    def test_loss_matches_independent_token_weighted_reference_not_window_average(self):
        model = ConstantModel()
        ids = torch.tensor([0] * 9 + [6], dtype=torch.long)
        log_normalizer = math.log(sum(math.exp(float(x)) for x in model.scores))
        reference = sum(
            log_normalizer - float(model.scores[int(target)]) for target in ids[1:]
        )
        for batch_size in (1, 2, 8):
            result = evaluate_model(model, ids, batch_size)
            self.assertEqual(result["next_token_targets"], 9)
            self.assertEqual(result["windows"], 3)
            self.assertAlmostEqual(result["nll_sum_nats"], reference, places=5)
            self.assertAlmostEqual(
                result["loss_nats_per_token"], reference / 9, places=6
            )
            self.assertAlmostEqual(
                result["perplexity"], math.exp(reference / 9), places=6
            )
            wrong_window_average = (2 * (log_normalizer - 5) + log_normalizer) / 3
            self.assertGreater(
                abs(result["loss_nats_per_token"] - wrong_window_average), 1
            )

    def test_evaluation_preserves_rng_and_mixed_module_training_flags(self):
        model = ConstantModel(draw_random=True).train()
        model.dropout.eval()
        torch.manual_seed(22)
        before = torch.get_rng_state().clone()
        evaluate_model(model, torch.tensor([0, 1, 2, 3, 4, 5]), 2)
        self.assertTrue(torch.equal(before, torch.get_rng_state()))
        self.assertTrue(model.training)
        self.assertFalse(model.dropout.training)

    def test_evaluation_restores_rng_and_mode_even_if_model_raises(self):
        model = ConstantModel(draw_random=True, fail=True).train()
        torch.manual_seed(23)
        before = torch.get_rng_state().clone()
        with self.assertRaisesRegex(RuntimeError, "deliberate"):
            evaluate_model(model, torch.tensor([0, 1, 2]))
        self.assertTrue(torch.equal(before, torch.get_rng_state()))
        self.assertTrue(model.training)

    def test_evaluation_cannot_change_next_dropout_training_step(self):
        config = ModelConfig(
            vocab_size=20, block_size=4, n_embd=8, n_head=2, n_layer=1, dropout=0.4
        )
        torch.manual_seed(31)
        model = TinyTransformer(config).train()
        ids = torch.arange(20)
        data_rng = torch.Generator().manual_seed(12)
        data_state = data_rng.get_state().clone()
        dropout_state = torch.get_rng_state().clone()
        x, y = sample_training_batch(ids, 4, 2, data_rng)
        logits, loss = model(x, y)
        torch.set_rng_state(dropout_state)
        data_rng.set_state(data_state)
        evaluate_model(model, ids, 2)
        x2, y2 = sample_training_batch(ids, 4, 2, data_rng)
        logits2, loss2 = model(x2, y2)
        self.assertTrue(torch.equal(x, x2))
        self.assertTrue(torch.equal(y, y2))
        torch.testing.assert_close(logits, logits2, rtol=0, atol=0)
        torch.testing.assert_close(loss, loss2, rtol=0, atol=0)

    def test_seeded_training_repeats_exactly_and_different_seed_differs(self):
        prepared = prepare_corpus(TEXT, 280)
        model = ModelConfig(
            vocab_size=prepared.tokenizer.get_vocab_size(),
            block_size=8,
            n_embd=8,
            n_head=2,
            n_layer=1,
            dropout=0.2,
        )
        config = RunConfig(steps=3, batch_size=2, evaluation_batch_size=3)
        first = run_seed(prepared, model, config, 7)
        torch.rand(99)
        second = run_seed(prepared, model, config, 7)
        different = run_seed(prepared, model, config, 8)
        self.assertEqual(first, second)
        self.assertNotEqual(
            first["model_state_sha256"], different["model_state_sha256"]
        )

    def test_summary_uses_sample_sd_and_keeps_single_seed_sd_unknown(self):
        values = [
            {
                name: {"loss_nats_per_token": value, "perplexity": math.exp(value)}
                for name in ("validation", "test")
            }
            for value in (1.0, 3.0, 5.0)
        ]
        result = summarize(values)["test"]["loss_nats_per_token"]
        self.assertEqual(result["mean"], 3)
        self.assertEqual(result["sample_standard_deviation"], 2)
        self.assertIsNone(
            summarize(values[:1])["test"]["loss_nats_per_token"][
                "sample_standard_deviation"
            ]
        )

    def test_invalid_inputs_fail_clearly(self):
        for text in ("", "a", "ab"):
            with self.subTest(text=text), self.assertRaises(ValueError):
                split_raw_text(text)
        with self.assertRaises(TypeError):
            split_raw_text(None)
        for vocab in (0, 255, True):
            with self.subTest(vocab=vocab), self.assertRaises(ValueError):
                fit_tokenizer("some training text", vocab)
        for kwargs in (
            {"steps": 0},
            {"batch_size": -1},
            {"learning_rate": float("nan")},
            {"gradient_clip": 0},
            {"weight_decay": -1},
            {"device": "other"},
        ):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                RunConfig(**kwargs)
        for ids in (
            torch.tensor([]),
            torch.tensor([0]),
            torch.tensor([-1, 0]),
            torch.tensor([[0, 1]]),
            torch.tensor([0.0, 1.0]),
        ):
            with self.subTest(ids=ids), self.assertRaises(ValueError):
                list(evaluation_windows(ids, 4))
        with self.assertRaises(ValueError):
            list(evaluation_windows(torch.arange(5), 0))
        with self.assertRaises(ValueError):
            evaluate_model(ConstantModel(), torch.tensor([0, 7]))
        with self.assertRaises(ValueError):
            sample_training_batch(torch.arange(8), 8, 1, torch.Generator())
        with self.assertRaises(ValueError):
            sample_training_batch(torch.arange(9), 8, 1, None)

    def test_cli_records_metadata_and_refuses_existing_output(self):
        with tempfile.TemporaryDirectory(
            prefix="transformer-evaluation-test-"
        ) as temporary:
            directory = Path(temporary)
            corpus = directory / "synthetic.txt"
            corpus.write_bytes(TEXT.encode())
            output = directory / "new-output"
            arguments = [
                "--corpus",
                str(corpus),
                "--output-dir",
                str(output),
                "--steps",
                "1",
                "--batch-size",
                "2",
                "--seeds",
                "7",
                "8",
                "--vocab-size",
                "280",
                "--embedding-dim",
                "8",
                "--heads",
                "2",
                "--layers",
                "1",
                "--context",
                "8",
                "--device",
                "cpu",
            ]
            with contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(main(arguments), 0)
            result_bytes = (output / "results.json").read_bytes()
            result = json.loads(result_bytes)
            self.assertEqual(result["corpus"]["sha256"], sha256(corpus.read_bytes()))
            self.assertEqual(
                result["tokenizer"]["sha256"],
                sha256((output / "tokenizer.json").read_bytes()),
            )
            self.assertEqual(result["seeds"], [7, 8])
            self.assertEqual(len(result["results"]), 2)
            self.assertEqual(
                len((output / "seed-results.jsonl").read_text().splitlines()), 2
            )
            self.assertNotIn(str(directory), result_bytes.decode())
            self.assertEqual(
                result["source_sha256"]["evaluate.py"],
                sha256((ROOT / "evaluate.py").read_bytes()),
            )
            self.assertEqual(
                sorted(p.name for p in output.iterdir()),
                [
                    "metadata.json",
                    "results.json",
                    "seed-results.jsonl",
                    "tokenizer.json",
                ],
            )
            with (
                contextlib.redirect_stderr(io.StringIO()),
                self.assertRaises(SystemExit),
            ):
                main(arguments)
            self.assertEqual((output / "results.json").read_bytes(), result_bytes)

    def test_cli_rejects_duplicate_seeds_without_creating_output(self):
        with tempfile.TemporaryDirectory(
            prefix="transformer-evaluation-test-"
        ) as temporary:
            output = Path(temporary) / "new-output"
            with (
                contextlib.redirect_stderr(io.StringIO()),
                self.assertRaises(SystemExit),
            ):
                main(["--output-dir", str(output), "--seeds", "7", "7"])
            self.assertFalse(output.exists())


if __name__ == "__main__":
    unittest.main()
