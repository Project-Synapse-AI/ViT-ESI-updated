# Training-throughput benchmark

Build and submit with the normal pinned-commit notebook workflow:

```powershell
python kaggle/push_job.py --job benchmark --title "vitesi throughput benchmark" --push
```

This is a short performance experiment, not a scientific accuracy result. It
regenerates the complete frozen `train8k_s0`, checks all 8,000 fingerprints,
and uses the same seed-0 6,000/2,000 split, depth-6 ViTESI, cosine loss,
Adam 0.001, 5 dB noise, batch 8, one visible GPU and original float64 loader
outputs as the calibration. It does not use the test set or change the
production training configuration.

Each configuration gets two fresh-model trials, eight warmup batches then
64 measured batches. Compare:

- FP32, workers 0, pinning off (current loader baseline).
- FP32, workers 0, pinning on.
- FP32, workers 2, pinning/persistent workers on.
- FP32, workers 4, pinning/persistent workers on.
- FP16 mixed precision using the fastest FP32 loading configuration.

Worker prefetch factor is 1 to bound queued dense targets in shared memory.
Worker startup and warmup are reported separately from measured throughput.
The output is `out/benchmark/benchmark.json` plus streamed `BENCH_RESULT`
and `BENCH_SUMMARY` lines. Results include samples/s, CPU loader-wait time,
CUDA-event transfer/cast and compute/optimizer times, peak allocated and
reserved VRAM, one-second GPU utilization/memory samples, finite losses and
FP16 scaler backoffs. CUDA event times overlap with CPU work and are not
additive wall-time fractions. Loader wait includes any prefetch wait and
is not a pure disk-I/O measurement. Too-short trials may have no GPU samples.

Two short repetitions are screening evidence, not full-epoch speed promises.
The custom training loop excludes Lightning logging, validation and checkpoint
overhead. It does not establish equal noise, equal convergence, or equal
accuracy. Any selected workers/precision configuration needs a matching
depth-6 calibration before A4 comparisons. Scaler backoffs should be reviewed
even if losses remain finite. Benchmark failure is a hard stop, not permission
to silently shrink a scientific run or fall back to CPU.

A local real-data CUDA plumbing check uses `--smoke` on an existing small
SEREEGA dataset (three batches/trial); it is not a Kaggle speed measurement:

```powershell
python kaggle/benchmark_training.py --simu-folder <mes_debug_python> --out <ignored-results-path> --smoke
```
