# Submit a Hardware Benchmark

This guide produces a privacy-limited, machine-checkable Leviathan benchmark submission. Matching the original CPU result is not guaranteed: CPU model, compiler, thread scheduling, thermals, memory, and background activity all affect timing.

## 1. Use clean pinned revisions

Clone the repository and check out the exact engine commit you intend to report. Do not benchmark with uncommitted engine changes.

```cmd
git clone https://github.com/ShiningSon/Leviathan-1.58bit-Engine.git
cd Leviathan-1.58bit-Engine
git checkout <FULL_ENGINE_COMMIT>
git status --short
```

Download the v09a package at its verified Hugging Face revision:

```cmd
hf download ShiningSon/Leviathan-MLGRU-200M-TinyStories-Instruct-v09a --revision 116a857bdaf2a1118d479d52aedba7e65cbff960 --local-dir leviathan_mlgru_200m_instruct_v09a
```

Install the repository requirements in your own environment. Model binaries and generated benchmark directories must remain untracked.

## 2. Prepare the machine

- Stop avoidable background work and record anything that cannot be stopped.
- Use wall power and a stable performance policy on mobile systems.
- Let the CPU cool before a long comparison and avoid changing thermal conditions between modes.
- Keep compiler, PyTorch, thread variables, model revision, engine commit, and prompt file identical across modes.
- Prefer at least 30 measured repeats for a canonical comparison; ten repeats are acceptable for an exploratory submission.
- Use `--mode-schedule interleave` so dense and sparse samples alternate after one warmup per mode.

## 3. Run dense and Top-K 0.10

For a dense-only smoke measurement:

```cmd
python scripts\benchmark_engine.py --model-dir .\leviathan_mlgru_200m_instruct_v09a --engine .\engine.py --prompts .\benchmarks\prompts_v02b_qa.json --architecture mlgru --prompt-template qa --max-new 80 --modes 0 --repeat 3 --mode-schedule interleave --model-repo ShiningSon/Leviathan-MLGRU-200M-TinyStories-Instruct-v09a --model-revision 116a857bdaf2a1118d479d52aedba7e65cbff960 --out-dir .\benchmark_runs\dense_smoke
```

For the comparable dense and Top-K `0.10` submission run:

```cmd
python scripts\benchmark_engine.py --model-dir .\leviathan_mlgru_200m_instruct_v09a --engine .\engine.py --prompts .\benchmarks\prompts_v02b_qa.json --architecture mlgru --prompt-template qa --max-new 80 --modes 0 0.10 --repeat 30 --warmup 1 --sparse-min-density 0.6 --no-top-k-sort --sparse-scope down --top-k-select histogram --mode-schedule interleave --model-repo ShiningSon/Leviathan-MLGRU-200M-TinyStories-Instruct-v09a --model-revision 116a857bdaf2a1118d479d52aedba7e65cbff960 --out-dir .\benchmark_runs\external_v09a_repeat30
```

Review `results.md` and `sample_outputs.txt`. Do not submit a run with process failures, parse failures, missing samples, or unreviewed QA failures.

## 4. Collect system information

The collector records only benchmark-relevant fields. It never collects username, hostname, IP address, MAC address, home directory, or local absolute paths. Unavailable optional values are written as `null`.

```cmd
python scripts\collect_system_info.py --json-out .\benchmark_runs\external_v09a_repeat30\hardware.json --architecture mlgru --model-repo ShiningSon/Leviathan-MLGRU-200M-TinyStories-Instruct-v09a --model-revision 116a857bdaf2a1118d479d52aedba7e65cbff960 --benchmark-command "python scripts\benchmark_engine.py --model-dir .\leviathan_mlgru_200m_instruct_v09a --engine .\engine.py --prompts .\benchmarks\prompts_v02b_qa.json --architecture mlgru --prompt-template qa --max-new 80 --modes 0 0.10 --repeat 30 --warmup 1 --sparse-min-density 0.6 --no-top-k-sort --sparse-scope down --top-k-select histogram --mode-schedule interleave --model-repo ShiningSon/Leviathan-MLGRU-200M-TinyStories-Instruct-v09a --model-revision 116a857bdaf2a1118d479d52aedba7e65cbff960 --out-dir .\benchmark_runs\external_v09a_repeat30"
```

Open `hardware.json` and confirm that you are comfortable sharing every value. Add compiler values manually if automatic detection is unavailable:

```cmd
python scripts\collect_system_info.py --compiler cl --compiler-version 19.40 --json-out .\benchmark_runs\external_v09a_repeat30\hardware.json
```

When overriding values, also repeat the model, revision, and benchmark-command arguments from the previous command.

## 5. Export and validate the submission

```cmd
python scripts\export_benchmark_submission.py --results .\benchmark_runs\external_v09a_repeat30\results.json --hardware-manifest .\benchmark_runs\external_v09a_repeat30\hardware.json --prompts .\benchmarks\prompts_v02b_qa.json --recommended-mode 0.10 --caveat "Measured on one local CPU; no general sparse speedup is claimed." --out .\benchmark_runs\external_v09a_repeat30\submission.json
python scripts\validate_benchmark_submission.py .\benchmark_runs\external_v09a_repeat30\submission.json
```

The exporter recalculates mean, median, population standard deviation, QA totals, sample counts, prompt hash, and output checksums. It rejects inconsistent or incomplete runs. Marking a mode as recommended also requires its QA rate to preserve or improve the dense rate.

## 6. Share the result

Post the result to the [`v0.9.1 cross-hardware benchmark submissions`](https://github.com/ShiningSon/Leviathan-1.58bit-Engine/issues/2) tracking issue using the **Hardware benchmark submission** template, and attach `submission.json` by dragging it into the issue. A small PR may add the JSON under a maintainer-approved location after review; do not commit `benchmark_runs/`, model weights, logs, or private environment files.

Maintainers will verify the schema, pinned revisions, QA preservation, caveats, and plausibility of the timing. Unverified numbers will not be added to canonical benchmark tables. A valid submission is evidence for that exact hardware and software configuration, not proof that Top-K is always faster.
