# Validation record

The September 2026 portfolio changes add a configurable module and small CPU
entry point. The historical notebook, report, data, and experiment outputs have
not been edited. The PC's additional uncommitted notebook edits remain in the
original checkout for separate review.

## Local checks

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

## Interpretation

Original validation scores and GPU-memory figures have not been reproduced in
this pass. The README identifies the original report as their source and notes
the experimental design limitations. No new performance or research claims are
based on the tiny demo. The original PC notebook remains unchanged.

A preliminary signature scan found no matching GitHub/provider token or private
key patterns in the inspected text files and history. Such a scan is not a
guarantee that arbitrary secrets or personal data are absent.
