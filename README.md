# G2 Turbo Engine (Pixel 7 Pro / Tensor G2)

An independent experimental project exploring Q4_0 GEMV kernels, compiled ARM
DOTPROD, and **real Android Vulkan GPU execution** on Mali-G710.

## v0.2: GPU-enabled Android APK

[Download APK from Releases](https://github.com/linkalls/g2-turbo-engine/releases)
· [Public Actions builds](https://github.com/linkalls/g2-turbo-engine/actions/workflows/android.yml)

The same APK measures:

- Scalar FP32 dequantization + GEMV (correctness reference).
- ARMv8.2 SDOT GEMV including blockwise INT8 activation quantization.
- A real Vulkan compute shader dispatch using Android's Vulkan 1.0 driver.
  GPU name, setup+upload time, cold dispatch wall time, and hot dispatch+fence
  mean/min/max times are returned with correctness checks. GPU may fail on some
  driver versions: error output remains available, and CPU results still work.

### PLaMo 3 Translate 0.8B shape presets

Official model config: hidden_size 1536, intermediate_size 6144,
num_hidden_layers 16, num_attention_heads 16, head_dim 128.
[Official config](https://huggingface.co/pfnet/plamo-3-translate-0.8b/blob/main/config.json)

- 768×1536 (v0.1 comparison).
- 1536×1536 (hidden-width proxy).
- 6144×1536 (MLP gate/up projection shape).
- 1536×6144 (MLP down projection shape).
- 2048×1536 (attention query projection shape, 16 heads × 128).

These use **deterministic synthetic Q4_0 values, not the model's weights**.
The APK does not load a GGUF, translate text, measure tokens/sec, or implement
an LLM execution graph. KleidiAI is **not yet linked**, so this is not yet a
four-way comparison. A real inference end-to-end comparison must follow.

### How to test

Install `g2-turbo-kernel-bench.apk` on your Pixel 7 Pro and press one of
the five size presets. You will see the CPU timing, GPU timing/error,
correctness results and raw JSON. Use the copy button to paste the JSON
back into your issue or ChatGPT conversation.

#### Measurement meaning

- GPU hot timing = wall clock from `vkQueueSubmit` through
  `vkWaitForFences`; **not GPU-only timestamp time**.
- GPU setup+upload includes shader read, device creation, pipeline creation,
  and mapped-buffer upload, but **excludes CPU repacking done before this step**.
- CPU DOTPROD timing includes the separately measured input quantization.
- All paths use the same deterministic weights and FP32 input. CPU SDOT path
  incurs lossy blockwise INT8 activation quantization, while GPU uses FP32.
- Fences and a compute-to-host barrier ensure correct readback.
- GPU correctness is validated against the scalar reference. Any error
  reported by the Vulkan driver **must not be interpreted as GPU acceleration**.
- Performance is specific to these kernels and does not imply an equal
  improvement in PLaMo translation throughput.

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
