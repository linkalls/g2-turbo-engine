# G2 Turbo Engine v0.4 (Pixel 7 Pro / Tensor G2)

An independent experimental project exploring Q4_0 GEMV kernels, compiled ARM
DOTPROD, and **real Android Vulkan GPU execution** on Mali-G710.

## v0.4: One-tap CPU pool + KleidiAI + Vulkan benchmark

[Download public APK from Releases](https://github.com/linkalls/g2-turbo-engine/releases)
· [GitHub Actions](https://github.com/linkalls/g2-turbo-engine/actions/workflows/android.yml)

Install the APK on Pixel 7 Pro. Tap **🔥 全5ケースを一括測定** just once.
The app automatically runs all five Q4_0 matrix sizes and retains partial results
even if an individual case fails. After completion, use **統合JSONをコピー**
or **統合JSONを共有** to provide the *entire batch* in one message.

A single JSON object contains:
- Device model, Android SDK, UTC start/end times and overall duration.
- Each of five matrix shapes with its own timings, accuracy and error status.
- Scalar reference GEMV and ARM DOTPROD for 1, 2, and 4 CPU threads
  (v0.3: worker threads are created/destroyed each call).
- Persistent 1/2/4-thread CPU worker pool: threads are created outside the
  timed kernel and reused between invocations. Tests both unpinned scheduling
  and best-effort affinity to the highest-ranked currently allowed cores.
  CPU rankings are based on readable cpu_capacity or cpufreq properties;
  device cpuset restrictions are never bypassed.
- Actual Arm KleidiAI v1.9.0 Q4_0 SDOT micro-kernel (not a fake comparison),
  including separately timed one-time RHS weight packing, per-inference LHS
  activation packing, and steady-state kernel execution. The upstream source
  is fetched at a pinned revision at build time (Apache-2.0 license).
- Three Vulkan GPU variants: legacy 64-lane, tiled 32-lane, and subgroup
  reduction. Their launch order is rotated between matrix shapes to limit
  fixed-order thermal drift.
- Per-variant GPU dispatch/fence wall time, optional Vulkan timestamp period and
  GPU timestamp execution time, cold timing, pipeline/setup/upload time,
  numerical RMSE and a correctness pass flag.
- If a Vulkan implementation is not supported, its error remains in the JSON.
  The other alternatives continue; the slowest case does not hide failures.

You can also tap one of the **individual measurement** buttons for a targeted test.

### PLaMo 3 Translate 0.8B shape presets

Official model config: hidden_size 1536, intermediate_size 6144,
num_hidden_layers 16, num_attention_heads 16, head_dim 128.
[Official config](https://huggingface.co/pfnet/plamo-3-translate-0.8b/blob/main/config.json)

| Name | Output rows | Input columns |
| --- | ---: | ---: |
| Legacy comparison | 768 | 1536 |
| Hidden-width proxy | 1536 | 1536 |
| MLP gate/up | 6144 | 1536 |
| MLP down | 1536 | 6144 |
| Attention Q (16 x 128) | 2048 | 1536 |

**Important limitations:** The APK uses deterministic synthetic Q4_0 matrices,
not PLaMo's actual model weights. It does not perform token generation, GGUF
loading, or inference graph execution. KleidiAI is now integrated as a standalone
Q4_0 SDOT kernel, not a full llama.cpp CPU backend.
The scalar reference is deliberately unoptimized. CPU thread costs include
creation and joining threads; a long-lived inference runtime might behave
differently. GPU wall time is vkQueueSubmit through vkWaitForFences,
while Vulkan timestamp (when supported) is the time between GPU TOP_OF_PIPE
and BOTTOM_OF_PIPE queries around the compute dispatch. Neither is a complete
end-to-end LLM benchmark. Driver support and numerical correctness must be
verified on the actual Pixel 7 Pro before drawing performance conclusions.

### Build architecture

- `native/src/kernel_bench.cpp`: scalar Q4_0, ARM DOTPROD and legacy CPU threading.
- `native/src/cpu_pool.cpp`: persistent workers, per-core affinity and CPU topology.
- `native/src/kleidi_bench.cpp`: actual pinned Arm KleidiAI Q4_0 pack and GEMV.
- `native/src/vulkan_bench.cpp`: Vulkan buffers, dispatch, timestamp queries,
  correctness and multi-variant comparison.
- `native/shaders/q4_0_gemv.comp`: baseline 64-lane shader.
- `native/shaders/q4_0_gemv_32.comp`: tiled32 vectorized shader.
- `native/shaders/q4_0_gemv_subgroup.comp`: subgroup arithmetic reduction.
- `app/src/main/java/dev/poteto/g2turbo/MainActivity.java`: one-tap automatic
  batch orchestration, live progress and aggregated JSON export.

### Security and reproducibility

Fully offline: no model download, network permission, or telemetry.
All code is MIT-licensed. APK uses a nonpersistent test signing certificate,
so this is not suitable for stable production upgrades. Reproduce builds with
the public [GitHub Actions workflow](.github/workflows/android.yml).

### Components

- `native/src/kernel_bench.cpp`: scalar, ARM SDOT, accuracy comparison.
- `native/src/vulkan_bench.cpp`: real Vulkan device, buffers, descriptors,
  SPIR-V pipeline, command recording, readback and cleanup.
- `native/shaders/q4_0_gemv.comp`: Q4_0 compute shader compiled to SPIR-V by CI.
- `app/src/main/java/dev/poteto/g2turbo/MainActivity.java`: Android UI,
  5 model-shape presets and clipboard output.

### Third-party code

The Arm KleidiAI project is copyright Arm Limited and licensed under Apache-2.0.
The upstream source is not vendored here; GitHub Actions checks out the
exact v1.9.0 commit via `scripts/fetch-kleidiai.sh`. License details:
https://github.com/ARM-software/kleidiai/blob/v1.9.0/LICENSES/Apache-2.0.txt
