import ast
import json
import math
import unittest
from dataclasses import asdict
from pathlib import Path

import torch
from tokenizers import Tokenizer
from torch import nn
from torch.nn import functional as F

from model import ModelConfig, TinyTransformer

ROOT = Path(__file__).resolve().parents[1]
torch.set_num_threads(1)


class TransformerTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(42)
        self.config = ModelConfig(
            vocab_size=40, block_size=8, n_embd=16, n_head=4, n_layer=2, dropout=0.0
        )
        self.model = TinyTransformer(self.config)

    def test_finite_backward_and_parameter_update(self):
        x = torch.randint(40, (2, 8))
        y = torch.randint(40, (2, 8))
        before = self.model.lm_head.weight.detach().clone()
        logits, loss = self.model(x, y)
        self.assertEqual(logits.shape, (2, 8, 40))
        self.assertTrue(torch.isfinite(loss))
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=1e-3)
        loss.backward()
        for parameter in self.model.parameters():
            self.assertIsNotNone(parameter.grad)
            self.assertTrue(torch.isfinite(parameter.grad).all())
        optimizer.step()
        self.assertFalse(torch.equal(before, self.model.lm_head.weight))

    def test_future_tokens_do_not_change_prefix_predictions(self):
        self.model.eval()
        x = torch.randint(40, (2, 8))
        changed = x.clone()
        changed[:, 4:] = (changed[:, 4:] + 1) % 40
        original_logits, _ = self.model(x)
        changed_logits, _ = self.model(changed)
        torch.testing.assert_close(original_logits[:, :4], changed_logits[:, :4])

    def test_matches_original_notebook_with_identical_weights(self):
        # Load only the six known class definitions from this trusted repository.
        # No data-fetching, training, checkpoint loading, or sweep cells execute.
        names = {
            "Head",
            "MultiHeadAttention",
            "RMSNorm",
            "FeedForward",
            "Block",
            "TinyTransformer",
        }
        notebook = json.loads((ROOT / "transformer.ipynb").read_text())
        definitions = []
        for cell in notebook["cells"]:
            if cell["cell_type"] != "code":
                continue
            tree = ast.parse("".join(cell["source"]))
            definitions.extend(
                node
                for node in tree.body
                if isinstance(node, ast.ClassDef) and node.name in names
            )
        self.assertEqual({node.name for node in definitions}, names)
        self.assertEqual(len(definitions), len(names))
        namespace = {
            "torch": torch,
            "nn": nn,
            "F": F,
            "math": math,
            **asdict(self.config),
        }
        exec(  # noqa: S102 - compare known class definitions from our trusted notebook
            compile(
                ast.Module(body=definitions, type_ignores=[]),
                "notebook-model-definitions",
                "exec",
            ),
            namespace,
        )
        original = namespace["TinyTransformer"]().eval()
        original.load_state_dict(self.model.state_dict(), strict=True)
        self.model.eval()
        x = torch.randint(40, (2, 8))
        y = torch.randint(40, (2, 8))
        old_logits, old_loss = original(x, y)
        new_logits, new_loss = self.model(x, y)
        torch.testing.assert_close(new_logits, old_logits, rtol=0, atol=0)
        torch.testing.assert_close(new_loss, old_loss, rtol=0, atol=0)

    def test_generation_crops_context_and_restores_training_mode(self):
        self.model.train()
        prompt = torch.randint(40, (1, 12))
        generated = self.model.generate(prompt, max_new_tokens=4)
        self.assertEqual(generated.shape, (1, 16))
        self.assertTrue(torch.equal(generated[:, :12], prompt))
        self.assertTrue(self.model.training)
        self.assertFalse(generated.requires_grad)

    def test_saved_bpe_tokenizer_round_trips_prompt(self):
        tokenizer = Tokenizer.from_file(str(ROOT / "tiny_shakespeare_bpe.json"))
        prompt = "O Romeo, Romeo! wherefore art thou Romeo?\n"
        ids = tokenizer.encode(prompt).ids
        self.assertEqual(tokenizer.decode(ids), prompt)
        self.assertLess(len(ids), len(prompt))

    def test_invalid_configuration_and_context_are_rejected(self):
        for kwargs in ({"n_embd": 15}, {"n_head": 3}, {"n_layer": 0}, {"dropout": 1}):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                ModelConfig(**kwargs)
        with self.assertRaises(ValueError):
            self.model(torch.zeros(1, 9, dtype=torch.long))
        with self.assertRaises(ValueError):
            self.model.generate(torch.zeros(1, 0, dtype=torch.long))


if __name__ == "__main__":
    unittest.main()
