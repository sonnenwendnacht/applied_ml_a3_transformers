# Validation record

The September 2026 portfolio changes add a configurable module and small CPU
entry point. The historical notebook, report, data, and experiment outputs have
not been edited. The PC's additional uncommitted notebook edits remain in the
original checkout for separate review.

## Initial portfolio baseline: local checks

- Python 3.14.0; PyTorch `2.10.0.dev20251018+cu130`; Tokenizers `0.22.2`.
  All checks below used CPU.
- Six tests pass: finite gradients and a real optimizer update, causal masking,
  exact equivalence with the notebook model under identical weights, generation
  beyond the context window with mode restoration, saved-tokenizer round trip,
  and configuration/context validation.
- `python demo.py --steps 5 --tokens 16` completed with finite losses and gradients.
  This is a functional smoke check, not a convergence result.
- The existing experiment-2 checkpoint loaded with strict state-dictionary
  matching and generated 32 new tokens using the saved BPE tokenizer. Weights
  were read locally, not committed or uploaded.

The six model class definitions are extracted from the trusted notebook only
inside the equivalence test. Data-fetching, training, sweep, and checkpoint-loading
cells are not executed by that test. Runtime inference uses `model.py` directly.

## Initial baseline interpretation

Original validation scores and GPU-memory figures have not been reproduced in
this pass. The README identifies the original report as their source and notes
the experimental design limitations. No new performance or research claims are
based on the tiny demo. The original PC notebook remains unchanged.

A preliminary signature scan found no matching GitHub/provider token or private
key patterns in the inspected text files and history. Such a scan is not a
guarantee that arbitrary secrets or personal data are absent.

## September 16: train-only tokenizer and held-out evaluation

The separate `evaluate.py` runner and 17 evaluation tests are AI-assisted
maintenance. The historical notebook, report, plots, samples, corpus, saved
tokenizer, `model.py`, and `demo.py` remain unchanged. This is not a retrospective
reproduction of the seven original configurations.

### Automated and independent checks

- All **23 tests** pass in both Python 3.12.3 / PyTorch 2.9.1+cpu and Python
  3.14.0 / PyTorch 2.10.0.dev20251018+cu130 (executed on CPU), with Tokenizers
  0.22.2. No CUDA execution or CUDA reproducibility claim is made.
- New tests cover raw Unicode boundaries, exact partition encoding, unseen
  byte/Unicode round trips, held-out changes not influencing tokenizer/training
  IDs, training batch bounds, independent data RNG, complete final evaluation
  windows, analytical token-weighted loss, model-mode/RNG restoration including
  exceptions, unchanged next dropout training step after evaluation, repeated
  seed equality, distinct-seed differences, sample SD, invalid inputs, output
  overwrite refusal, nested output directories, and fresh-process tokenizer
  consistency.
- An independent CPU script checked 306 length/context combinations and 918
  batched coverage combinations, plus four analytical-loss batch sizes and
  separate leakage/RNG/repetition checks. Its two fresh fixture CLI processes
  reproduced all values except timestamps. These checks used the Python 3.14
  environment; the recorded full experiment below used Python 3.12.
- The complete-corpus tokenizer and training-token hashes matched across three
  additional fresh Python 3.12 processes.
- Ruff 0.16.7 lint/format checks pass for the two new Python files. Git whitespace
  checks pass. CI runs the complete suite, the existing CPU demo, and a separate
  two-step evaluation on a synthetic fixture; CI is not the full training sweep.

### Recorded full experiment and reproduction

```bash
python evaluate.py --steps 1000 --batch-size 16 --seeds 17 42 2026 --device cpu --output-dir evaluation-runs/heldout
python evaluate.py --steps 1000 --batch-size 16 --seeds 17 42 2026 --device cpu --output-dir evaluation-runs/reproduction
```

Both commands used the same pinned direct dependencies in
`requirements-evaluation.txt`, CPU with one thread, and the bundled corpus.
The second complete run matched the first run's 3,000 step losses, all held-out
metrics, final model-state hashes, tokenizer bytes, and metadata except the
creation timestamp. Full source, corpus, partition, tokenizer, and environment
provenance are in `results/heldout-2026-09-16/results.json`; exact file hashes and
comparison details are in the adjacent reproduction record.

Mean test loss: **3.585119 ± 0.007388** nats/token (sample SD across three seeds).
Mean test perplexity: **36.058291 ± 0.266924**. Fixed-final-step weights were used
for every seed, not the best test or validation checkpoint. All 60,347 test
next-token targets were scored once, including the final short window.

The result is scoped to one sequential corpus split and a small fixed training
budget. It does not establish broad benchmark performance, cross-version or
cross-hardware bitwise reproducibility, document-level independence, or a fair
comparison against the historical losses with a different tokenizer/split.

Gitleaks 8.30.1 found no signatures in the prior retained history or new source
before result artifacts were added. The final tree scan flagged four generic-key
matches: the train/validation/test token-ID SHA-256 fields and the tokenizer
SHA-256 field in the reproduction record. These were checked against freshly
computed public-corpus/tokenizer hashes and identified as checksum false
positives, not credentials; no broad ignore rule was added. Scanning is not
proof that all possible secrets are absent. Only selected source, a synthetic
fixture, and public-corpus results are included; no model checkpoint or personal
data is added.
