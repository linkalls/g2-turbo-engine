# G2 Turbo Engine v0.3 (Pixel 7 Pro / Tensor G2)

An independent experimental project exploring Q4_0 GEMV kernels, compiled ARM
DOTPROD, and **real Android Vulkan GPU execution** on Mali-G710.

## v0.3: One-tap complete benchmark

[Download public APK from Releases](https://github.com/linkalls/g2-turbo-engine/releases)
· [GitHub Actions](https://github.com/linkalls/g2-turbo-engine/actions/workflows/android.yml)

Install the APK on Pixel 7 Pro. Tap **🔥 全5ケースを一括測定** just once.
The app automatically runs all five Q4_0 matrix sizes and retains partial results
even if an individual case fails. After completion, use **統合JSONをコピー**
or **統合JSONを共有** to provide the *entire batch* in one message.

A single JSON object contains:
- Device model, Android SDK, UTC start/end times and overall duration.
- Each of five matrix shapes with its own timings, accuracy and error status.
- Scalar reference GEMV, ARM DOTPROD for 1, 2, and 4 CPU threads.
- Three Vulkan GPU variants: legacy 64-lane, tiled 32-lane, and subgroup reduce.
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
loading, or inference graph execution. KleidiAI is not integrated yet.
The scalar reference is deliberately unoptimized. CPU thread costs include
creation and joining threads; a long-lived inference runtime might behave
differently. GPU wall time is vkQueueSubmit through vkWaitForFences,
while Vulkan timestamp (when supported) is the time between GPU TOP_OF_PIPE
and BOTTOM_OF_PIPE queries around the compute dispatch. Neither is a complete
end-to-end LLM benchmark. Driver support and numerical correctness must be
verified on the actual Pixel 7 Pro before drawing performance conclusions.

### Build architecture

- `native/src/kernel_bench.cpp`: scalar Q4_0, ARM DOTPROD and CPU threading.
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
