# A2: fixed averaging instead of learned attention

Tracking: issue #8. Initial scope: training seed 0, six encoder layers, global
mean and local-window averaging. Keep the frozen A4 workers4/FP32 protocol,
all 100 epochs, the 6000/2000 split, and both 2000-sample five-metric evaluations.
Temporal error (ms) remains a primary reported metric alongside LE, AUC, nMSE
and PSNR. No test result is used for tuning or model selection.

## Chosen interventions

- `-vit_mixer attention` is the unchanged default original architecture.
- `-vit_mixer mean` replaces each attention branch with the mean of all
  normalized timepoint tokens, broadcast back to each token.
- `-vit_mixer window -vit_window_radius 5` uses an equally weighted mean over
  the current token and up to five neighbours on either side. This is a
  prespecified local alternative, not a tuned window. At the boundaries,
  divide by the number of real tokens, excluding padding.
- Keep positional embeddings, pre-LayerNorms, residuals, feed-forward blocks,
  dropout and input/output projections. Do not widen feed-forward blocks;
  report the reduced parameter count and reserve capacity matching for A5.
- Averaging has no Q/K/V/output projections or attention-weight dropout.
  Retain the attention branch's output dropout. This removes the attention
  sublayer as a whole; it does not isolate only its content-dependent weights.
- Construct the original encoder first, then replace the branches without
  consuming RNG. Shared non-attention weights have identical initialization
  under the same seed. Subsequent training noise trajectories need not match.

The original six-layer matched Kaggle control from A4 is available (source
2f4068e, weights532cbeee..., heldout/testLE7.03977/6.96316mm). Reuse requires
proof that default-attention initialization/forward/state-dict behaviour is
unchanged and a recorded cross-source audit; do not run the A4 comparison
script on these variants or mix in the older local reference.

## Execution gates

1. Verify default attention against the pre-A2 implementation under fixed RNG,
   and independently check averaging, boundary handling, gradients and counts.
2. Push a frozen source commit; smoke each averaging variant using real frozen
   data on Kaggle before its full run. Confirm correct training/eval mixer,
   finite five metrics, split, profile, selected-weight provenance and hashes.
3. Submit the two full100-epoch jobs under the user's authorization to begin A2,
   using matching train/eval mixer flags and the unchanged production profile.
4. Audit both full outputs and cross-source control compatibility, bootstrap
   paired sample deltas, archive with read-back checksums, update issue/PR/vault.

Initial quota observed16.41GPUh, sufficient for this two-variant initial scope
using A4's approximately4.2-4.5h measured training durations as rough context,
not a duration guarantee. Check fresh quota before submissions. Additional
seeds, windows or architectures require a separate scope/budget decision.
