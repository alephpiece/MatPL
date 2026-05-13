# Profiling

MatPL can collect PyTorch profiler traces during training. Profiling is useful
for finding slow Python steps, PyTorch operator overhead, custom op time, GPU
kernel time, and CPU/GPU synchronization.

Add a `profiling` block to the input JSON to enable it.

## Example

```json
{
  "profiling": {
    "enabled": true,
    "activities": [
      "cpu",
      "gpu"
    ],
    "record_shapes": true,
    "profile_memory": false,
    "with_stack": false,
    "with_flops": false,
    "schedule": {
      "skip_first": 100,
      "wait": 0,
      "warmup": 50,
      "active": 100,
      "repeat": 1
    },
    "trace": {
      "enabled": true,
      "dir_name": "profiling",
      "worker_name": "dtk",
      "use_gzip": true
    }
  }
}
```

With this configuration, MatPL skips the first 100 training steps, warms up for
50 steps, records 100 steps, and writes compressed profiler trace files under
the `profiling` directory.

## Viewing Results

Trace files use PyTorch profiler's TensorBoard trace format:

```text
*.pt.trace.json
*.pt.trace.json.gz
```

Open them with TensorBoard:

```bash
tensorboard --logdir profiling
```

The trace JSON can also be opened by Perfetto.

`trace.dir_name` is resolved relative to the directory where MatPL is executed.
Absolute paths are also accepted.

## Quick Enable

For a quick profiling run, this is enough:

```json
{
  "profiling": true
}
```

This enables profiling with default settings and writes traces to
`profiler/trace`.

To disable profiling:

```json
{
  "profiling": false
}
```

Missing `profiling` also means disabled.

## Schedule

The schedule is counted in profiler steps. In MatPL training, one profiler step
corresponds to one training batch/update step, not one epoch.

```json
{
  "schedule": {
    "skip_first": 100,
    "wait": 0,
    "warmup": 50,
    "active": 100,
    "repeat": 1
  }
}
```

Fields:

- `skip_first`: skip this many initial training steps before profiling starts.
- `wait`: wait this many steps at the start of each profiling cycle.
- `warmup`: run profiler warmup for this many steps. Warmup data is not written
  to the trace.
- `active`: record this many steps and then write one trace.
- `repeat`: repeat the `wait -> warmup -> active` cycle this many times.

For long epochs, avoid recording too many steps. Large traces can slow training
and become difficult to open. A practical first run is:

```json
{
  "skip_first": 20,
  "wait": 0,
  "warmup": 5,
  "active": 10,
  "repeat": 1
}
```

For a longer, more stable sample, use a larger warmup and active window, such
as the example at the top of this page.

## Trace Options

```json
{
  "trace": {
    "enabled": true,
    "dir_name": "profiling",
    "worker_name": "dtk",
    "use_gzip": true
  }
}
```

- `enabled`: whether to write trace files.
- `dir_name`: output directory for trace files.
- `worker_name`: name included in trace file names. Use a rank, host, or
  backend name when comparing runs.
- `use_gzip`: write gzip-compressed traces. This is recommended for longer
  traces.

If `profiling.enabled` is true and no output block is provided, `trace` is
enabled by default.

## Activities

```json
{
  "activities": ["cpu", "gpu"]
}
```

- `cpu`: collect CPU-side profiler events.
- `gpu`: collect GPU activity if the PyTorch runtime has a GPU backend.

Use `gpu` in MatPL input files. PyTorch uses the CUDA profiler activity name
for both CUDA and ROCm/HIP builds, so MatPL maps `gpu` to the correct PyTorch
profiler activity internally. On CPU-only runtime, GPU activity is ignored.

## Extra Profiler Options

```json
{
  "record_shapes": true,
  "profile_memory": false,
  "with_stack": false,
  "with_flops": false
}
```

- `record_shapes`: record tensor shapes. Useful for operator analysis.
- `profile_memory`: collect memory profiling data.
- `with_stack`: collect Python stack traces. This can add noticeable overhead.
- `with_flops`: estimate FLOPs for supported operators.

Defaults:

```json
{
  "record_shapes": true,
  "profile_memory": false,
  "with_stack": false,
  "with_flops": false
}
```

## Summary Table

MatPL can optionally print PyTorch profiler averages to stdout:

```json
{
  "summary": {
    "enabled": true,
    "sort_by": "gpu_time_total",
    "row_limit": 30
  }
}
```

`summary` is disabled by default. Preparing the averages table can be expensive
for long traces, so use trace files as the default profiling output.

Common `sort_by` values:

- `gpu_time_total`
- `self_gpu_time_total`
- `cpu_time_total`
- `self_cpu_time_total`

`gpu_time_total` and `self_gpu_time_total` are MatPL-friendly names. They map to
PyTorch's CUDA profiler sort keys internally, including on ROCm/HIP builds.

## Notes

- Profiling adds overhead. Use short `active` windows first.
- `wait`, `warmup`, and `active` are training steps, not epochs.
- Trace output can be large. Use `use_gzip: true` for longer runs.
- Profiling is currently integrated in the Python training flow.
