# Tiny Shakespeare Transformer Experiments

A PyTorch causal language model with byte-pair tokenization, sinusoidal positions,
multi-head attention, and pre-normalization with RMSNorm. The original course
study explored seven width/head/depth configurations on Tiny Shakespeare; the
maintained evaluator tests one fixed architecture across three training seeds
with tokenization fitted on training text only.

The original work was completed for an applied machine-learning course in April
2026 by Junzhe Zong. The notebook, report, and historical outputs are preserved.
The standalone model, CPU demo, and tests were added during portfolio cleanup in
September 2026 with AI assistance.

[CPU demo](#run-a-small-cpu-demo) · [Held-out results](#recorded-follow-up-results) ·
[Historical study](#historical-course-experiments) · [Validation](VALIDATION.md)

The maintained experiment scores every held-out target token and records
source, data, tokenizer and model-state hashes. A full repeat matched the three
runs exactly in the recorded CPU environment; this is reproducibility evidence,
not an external language-model benchmark.

## Run a small CPU demo

From the repository root, use Python 3.11 or newer. The demo reads the bundled text and tokenizer;
it does not download a model or start the seven-experiment sweep.

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install 'torch>=2.6,<3' --index-url https://download.pytorch.org/whl/cpu
python -m pip install -r requirements.txt
python demo.py
python -m unittest discover -s tests -v
```

The default demo runs five CPU training steps and generates 32 tokens. This checks
that batching, loss calculation, backpropagation, and sampling work. Its output
is not intended to be fluent or to reproduce the historical validation scores.

## Train-only tokenizer and held-out evaluation

The September 16, 2026 maintenance pass adds a separate evaluation runner. It
addresses the historical tokenizer's exposure to validation text without
rewriting the original notebook or retroactively changing its reported scores.
This implementation, its tests, and the new evaluation are AI-assisted follow-up
work, not the original course experiment.

For the recorded CPU environment, use Python 3.12 and install:

```bash
python -m pip install 'torch==2.9.1' --index-url https://download.pytorch.org/whl/cpu
python -m pip install -r requirements-evaluation.txt
python evaluate.py --steps 1000 --batch-size 16 --seeds 17 42 2026 --device cpu --output-dir evaluation-runs/heldout
```

Choose a new output directory for every run; the runner refuses to overwrite an
existing directory. It does not download data, call a model API, or load a saved
checkpoint. The bundled corpus is sufficient.

Evaluation protocol:

- Split **raw text first** into sequential 80% training, 10% validation, and 10%
  test partitions. Train a new byte-level BPE tokenizer on training text only;
  encode each partition separately. A complete byte alphabet handles characters
  absent from training without fitting on held-out text.
- Train the same existing Transformer implementation: 64-dimensional embeddings,
  four attention heads, two layers, context length 64, and 163,392 parameters.
  The follow-up uses final-step weights, with no validation/test-based checkpoint
  selection. The tokenizer is shared across the three training seeds. This is
  not the four-layer architecture used by the checkpoint-generation example.
- Keep training minibatch sampling independent of model randomness. Evaluate
  fixed, nonoverlapping context windows and include the final short window;
  weight losses by their number of target tokens rather than averaging windows.
- Record corpus/split/tokenizer/source hashes, configuration, runtime versions,
  per-seed metrics, and aggregate mean/sample standard deviation. No model
  weights or private data are uploaded.

This is a small controlled evaluation, not a rerun of the seven-architecture
course sweep. Sequential corpus partitions can share phrases, characters, and
literary sources; disjoint positions are not an independent-domain benchmark.
Windowed evaluation resets context at each block, and perplexity depends on this
tokenizer. Do not compare its numbers directly with the historical table below.

### Recorded follow-up results

CPU, one thread, Python 3.12.3, PyTorch 2.9.1+cpu, Tokenizers 0.22.2. Each seed
trained for 1,000 steps with batch size 16; all 59,004 validation and 60,347 test
next-token targets were evaluated. Loss is natural-log cross-entropy per token.

| Training seed | Validation loss | Test loss | Test perplexity |
| --- | ---: | ---: | ---: |
| 17 | 3.579705 | 3.593603 | 36.364864 |
| 42 | 3.584358 | 3.581643 | 35.932536 |
| 2026 | 3.584388 | 3.580110 | 35.877472 |
| Mean ± sample SD | 3.582817 ± 0.002695 | 3.585119 ± 0.007388 | 36.058291 ± 0.266924 |

The SD describes variability across these three training seeds, not uncertainty
across datasets or splits. A second complete three-seed run matched all 3,000
training losses, every held-out metric, tokenizer bytes, and final model-state
hashes exactly in this CPU environment. Only run timestamps differed.

See [full measurements and provenance](results/heldout-2026-09-16/results.json),
the [train-only tokenizer](results/heldout-2026-09-16/tokenizer.json), and the
[reproduction record](results/heldout-2026-09-16/reproduction.json). The original
full-corpus tokenizer remains at the repository root solely for the historical
checkpoint/demo path.

## Generate from an original checkpoint

Weights are excluded from Git. If you have the experiment-2 state dictionary:

```bash
python demo.py --checkpoint /path/to/exp2_64d_4h_4L_weights.pth --prompt 'O Romeo, Romeo! ' --tokens 64
```

The default configuration matches experiment 2: embedding dimension 64, four
heads, four layers, context 64, and the bundled tokenizer's 500-token vocabulary.
For another checkpoint, specify `--embedding-dim`, `--heads`, `--layers`, and
`--context` to match its training configuration and supply the corresponding
tokenizer with `--tokenizer`. Changing the tokenizer can change token meanings
even when the vocabulary size matches.

The demo uses the saved BPE tokenizer for encoding and decoding and loads only
the checkpoint's weights on CPU. The model keeps the original state-dictionary
names, and a test compares its logits and loss with the notebook model using
identical weights.

## Historical course experiments

All seven historical runs used 2,000 training steps, context length 64, batch
size 32, and learning rate `1e-3`. Embedding dimensions ranged from 64 to 512,
heads from 4 to 32, and depth from 1 to 128 layers.

| Configuration | Embedding | Heads | Layers | Reported validation loss |
| --- | ---: | ---: | ---: | ---: |
| Experiment 1 | 64 | 4 | 2 | 3.7020 |
| Experiment 2 | 64 | 4 | 4 | 3.6241 |
| Experiment 4 | 512 | 8 | 16 | 4.4122 |
| Experiment 6 | 512 | 8 | 32 | 4.8123 |
| Experiment 7 | 512 | 32 | 128 | 4.6464 |

These are values reported in [the original course report](aml_a3.pdf), not
measurements rerun during cleanup. Experiment 2 has the lowest validation loss
among the five configurations tabulated in that report; results for experiments
3 and 5 are not included in its summary table. Saved plots and text samples
exist for all seven configurations.

![Experiment 2 training and validation loss](exp2_64d_4h_4L_loss.png)

The observed runs suggest that increasing capacity under this fixed training
setup did not reliably improve validation loss. They do not establish a universal
optimal architecture or prove a specific cause of instability.

## Limits and next experiments

- The historical tokenizer was fitted on the full corpus before the sequential
  80/20 split. The new runner fixes this boundary for its separate follow-up;
  the historical seven-architecture comparisons have not been retokenized or
  rerun and retain that limitation.
- The historical comparisons lack repeated seeds and uncertainty estimates.
  Architecture changes share a fixed learning rate and step budget, not an
  equal compute budget or individually tuned optimizers.
- Loss curves and one attention head do not by themselves establish gradient
  explosion or causal explanations for instability. Those interpretations in
  the historical report need gradient statistics and controlled experiments.
- Generated examples are qualitative. No external language-model benchmark or
  downstream deployment is claimed.
- The historical notebook uses globals and CUDA-specific telemetry. Its final
  sweep includes a 128-layer run and is not a quick CPU demonstration.

## Repository guide

| File | Purpose |
| --- | --- |
| `model.py` | Configurable model with explicit validation; no notebook globals |
| `demo.py` | Small CPU training check and checkpoint-based text generation |
| `evaluate.py` | Train-only BPE and reproducible multi-seed held-out evaluation |
| `tests/test_evaluation.py` | Split boundaries, tokenizer isolation, target accounting, and reproducibility |
| `results/heldout-2026-09-16/` | New CPU measurements, train-only tokenizer, and reproduction evidence |
| `tests/test_model.py` | Causal masking, gradients, notebook equivalence, generation, tokenizer checks |
| `transformer.ipynb` | Original notebook and experiment sweep |
| `aml_a3.pdf` | Historical course report |
| `exp*_loss.png`, `exp*_attention.png`, `exp*_sample.txt` | Historical experiment outputs |
| `tiny_shakespeare_bpe.json` | Saved byte-level BPE tokenizer |

For the notebook environment, install `requirements-notebook.txt` and open
`transformer.ipynb` in JupyterLab. Review the sweep cell before executing it.

## Credits

Tiny Shakespeare comes from [Andrej Karpathy's char-rnn dataset](https://github.com/karpathy/char-rnn/tree/master/data/tinyshakespeare).
For a related educational GPT-style implementation, see
[Karpathy's language-model tutorial](https://github.com/karpathy/ng-video).
This course version uses RMSNorm and sinusoidal positions. Dependencies include
[PyTorch](https://pytorch.org/) and [Hugging Face Tokenizers](https://huggingface.co/docs/tokenizers/).
The original report contains its AI-tool disclosure. No repository-wide license
has been added to the pre-existing course materials.
