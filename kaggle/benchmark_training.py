"""Short throughput experiment on real frozen data; not a convergence experiment."""

import argparse
import gc
import json
import math
import os
from pathlib import Path
import statistics
import subprocess
import sys
import threading
import time

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "model_training"))

import numpy as np
import torch
from torch.utils.data import DataLoader, random_split
from pytorch_lightning import seed_everything
from loaders import EsiDatasetds_new
from models.vit import EEGViT
from utils.utl import CosineSimilarityLoss


def sample_gpu(stop, samples):
    while not stop.wait(1):
        try:
            p = subprocess.run([
                "nvidia-smi", "-i", "0", "--query-gpu=utilization.gpu,memory.used",
                "--format=csv,noheader,nounits"], capture_output=True, text=True, timeout=5)
            if p.returncode == 0:
                util, memory = map(float, p.stdout.strip().split(","))
                samples.append({"util_percent": util, "used_mib": memory})
        except (OSError, ValueError, subprocess.TimeoutExpired):
            pass


def trial(dataset, workers, pin, precision, steps, warmup, repeat):
    seed_everything(0, workers=True)
    train, _ = random_split(dataset, [int(len(dataset) * .75), len(dataset) - int(len(dataset) * .75)])
    kwargs = dict(num_workers=workers, pin_memory=pin, persistent_workers=workers > 0)
    if workers:
        kwargs["prefetch_factor"] = 1
    loader = DataLoader(train, batch_size=8, shuffle=True, **kwargs)
    model = EEGViT(75, 994).cuda().train()
    optimizer = torch.optim.Adam(model.parameters(), lr=.001)
    criterion = CosineSimilarityLoss()
    scaler = torch.amp.GradScaler("cuda", enabled=precision == "16-mixed")
    started = time.perf_counter()
    iterator = iter(loader)
    init_s = time.perf_counter() - started
    losses, events, waits = [], [], []

    def step():
        t = time.perf_counter()
        eeg, src = next(iterator)
        waits.append(time.perf_counter() - t)
        a, b, c = [torch.cuda.Event(enable_timing=True) for _ in range(3)]
        a.record()
        # Match main_train: transfer loader float64, then cast in model step.
        eeg = eeg.to("cuda", non_blocking=pin).float()
        src = src.to("cuda", non_blocking=pin).float()
        b.record()
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast("cuda", dtype=torch.float16, enabled=precision == "16-mixed"):
            prediction = model(eeg)
            loss = criterion(prediction, src)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        c.record()
        losses.append(loss.detach())
        events.append((a, b, c))

    warm_start = time.perf_counter()
    for _ in range(warmup):
        step()
    torch.cuda.synchronize()
    warm_s = time.perf_counter() - warm_start
    losses.clear()
    events.clear()
    waits.clear()
    torch.cuda.reset_peak_memory_stats()
    samples, stop = [], threading.Event()
    monitor = threading.Thread(target=sample_gpu, args=(stop, samples), daemon=True)
    monitor.start()
    scale_start = scaler.get_scale()
    started = time.perf_counter()
    try:
        for i in range(steps):
            step()
            if (i + 1) % 16 == 0:
                print(f"BENCH_PROGRESS workers={workers} pin={pin} precision={precision} repeat={repeat} batch={i+1}/{steps}", flush=True)
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - started
    finally:
        stop.set()
        monitor.join(timeout=6)
    values = torch.stack(losses).float().cpu().numpy()
    if not np.isfinite(values).all():
        raise RuntimeError("HARD STOP: non-finite benchmark losses")
    if not all(torch.isfinite(p).all().item() for p in model.parameters()):
        raise RuntimeError("HARD STOP: non-finite model parameters")
    result = dict(workers=workers, pin_memory=pin, persistent_workers=workers > 0,
                  precision=precision, repeat=repeat, batch_size=8, steps=steps,
                  iterator_init_s=init_s, warmup_s=warm_s, measured_s=elapsed,
                  samples_per_s=steps * 8 / elapsed,
                  loader_wait_s=sum(waits),
                  mean_transfer_ms=statistics.mean(a.elapsed_time(b) for a, b, _ in events),
                  mean_gpu_step_ms=statistics.mean(b.elapsed_time(c) for _, b, c in events),
                  peak_allocated_mib=torch.cuda.max_memory_allocated() / 2**20,
                  peak_reserved_mib=torch.cuda.max_memory_reserved() / 2**20,
                  amp_scale_start=scale_start, amp_scale_end=scaler.get_scale(),
                  amp_backoffs=max(0, round(math.log2(scale_start / scaler.get_scale()))),
                  loss_mean=float(values.mean()), losses_finite=True,
                  gpu_samples=samples,
                  mean_gpu_util_percent=statistics.mean(s["util_percent"] for s in samples) if samples else None)
    # Drop the iterator before loader so persistent subprocesses shut down here.
    del iterator, loader, model, optimizer, scaler, losses, events
    gc.collect()
    torch.cuda.empty_cache()
    print("BENCH_RESULT " + json.dumps(result), flush=True)
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--simu-folder", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--smoke", action="store_true", help="local plumbing check only")
    args = p.parse_args()
    if not torch.cuda.is_available():
        raise SystemExit("HARD STOP: CUDA required")
    if torch.cuda.device_count() != 1:
        raise SystemExit("HARD STOP: exactly one visible GPU required")
    folder = Path(args.simu_folder).resolve()
    name = folder.name
    dataset = EsiDatasetds_new(str(folder.parents[4]),
        str(folder / f"{name}fsav_994_config.json"), name, "fsav_994", "standard_1020",
        100 if args.smoke else 8000, 5, {"white": 1., "pink": 0.}, norm="linear")
    if not args.smoke and len(dataset) != 8000:
        raise SystemExit("HARD STOP: benchmark needs full frozen train8k_s0")
    steps, warmup, repeats = (2, 1, 1) if args.smoke else (64, 8, 2)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    record = dict(status="running", purpose="throughput only; not accuracy or equivalence",
                  torch=torch.__version__, gpu=torch.cuda.get_device_name(),
                  cpu_count=os.cpu_count(), torch_threads=torch.get_num_threads(),
                  git_commit=subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
                  dataset=str(folder), warmup_batches=warmup, measured_batches=steps,
                  repeats=repeats, trials=[])

    def save():
        (out / "benchmark.json").write_text(json.dumps(record, indent=2) + "\n")

    save()
    def run(workers, pin, precision):
        scores = []
        for repeat in range(repeats):
            result = trial(dataset, workers, pin, precision, steps, warmup, repeat)
            record["trials"].append(result)
            save()
            scores.append(result["samples_per_s"])
        return statistics.median(scores)
    try:
        baseline = run(0, False, "32-true")
        choices = [(0, True), (2, True), (4, True)]
        scores = [run(w, pin, "32-true") for w, pin in choices]
        # Retain the baseline if all loading changes are slower.
        candidates = [(baseline, 0, False)] + [(s, w, pin) for s, (w, pin) in zip(scores, choices)]
        best, workers, pin = max(candidates)
        amp = run(workers, pin, "16-mixed")
        record.update(status="passed", baseline_samples_per_s=baseline,
                      best_fp32=dict(workers=workers, pin_memory=pin, samples_per_s=best, speedup=best / baseline),
                      fp16=dict(workers=workers, pin_memory=pin, samples_per_s=amp, speedup=amp / baseline),
                      caveat="Two short repeats, warm caches, no validation/checkpoint overhead; finite losses do not establish convergence. Worker changes alter noise RNG. Recalibrate chosen settings.")
    except BaseException as error:
        record.update(status="failed", error=f"{type(error).__name__}: {error}")
        raise
    finally:
        save()
    print("BENCH_SUMMARY " + json.dumps({k: record[k] for k in ["baseline_samples_per_s", "best_fp32", "fp16"]}), flush=True)


if __name__ == "__main__":
    main()
