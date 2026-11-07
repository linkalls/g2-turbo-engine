# G2 Turbo Engine

**Public Tensor G2 / Pixel 7 Pro research project** for low-level LLM inference kernels.

## Current deliverable: one installable Android APK

The first public APK is a **Q4_0 matrix-vector multiply microbenchmark**. It is
not a whole LLM, not a PLaMo translation app, and **does not invoke the GPU**.
Its purpose is to measure a portable Q4_0 FP32 reference against ARM SDOT
with blockwise INT8 activation quantization on a real Tensor G2 device.

### Build and download

- [Latest GitHub Actions builds](https://github.com/linkalls/g2-turbo-engine/actions/workflows/android.yml)
- Select a successful **Public Android benchmark APK** workflow run.
- Open **Artifacts** and download **g2-turbo-kernel-bench-apk**.
- Unzip the artifact, install `g2-turbo-kernel-bench.apk` on Pixel 7 Pro.
- Tap **768 × 1536 のGEMVを測定**, then compare times and accuracy in the JSON result.
- [Public Releases](https://github.com/linkalls/g2-turbo-engine/releases)

No model download, internet permission, telemetry or analytics is present.

### Bench measurements

The app reports **reference_ms**, **dotprod_ms**,
**activation_quantization_ms**, **speedup_including_quantization**,
**normalized_rmse**, **max_absolute_error**, checksums and **pass**.
On Tensor G2, `arm_dotprod_compiled` should be `true`.

The results measure one kernel only, not model loading, prompt prefill,
multi-layer graph execution, KV cache, or tokens/second. Do not compare
its speedup directly with PLaMo translation latency.

### Roadmap

1. Validate on Pixel 7 Pro and collect multiple cold/hot measurements.
2. Measure Q4_0 and Q4_K variants, CPU-thread behavior and memory bandwidth.
3. Integrate Vulkan buffers + dispatch and verify the experimental Q4_0
   shader against the CPU golden output on Mali-G710.
4. Replace naïve workgroup reduction with subgroup/tiled variants as supported.
5. Integrate proven kernels with a real GGUF inference runtime for end-to-end
   PLaMo benchmark; measure tok/s, time-to-first-token, energy, temperature.

The first GPU shader is in `native/shaders/q4_0_gemv.comp`. GitHub Actions
only validates its SPIR-V syntax: **the APK does not execute this shader**.

### Host CPU validation

```bash
cmake -S native -B build/native -DCMAKE_BUILD_TYPE=Release
cmake --build build/native
./build/native/g2-kernel-bench 768 1536 12
```

### Project structure

```text
app/                                  Android Java UI
native/src/kernel_bench.cpp           C++ Q4_0 reference and ARM SDOT
native/shaders/q4_0_gemv.comp          experimental Vulkan shader
.github/workflows/android.yml         public reproducible APK build
```

The public APK uses a test/debug signing key. Do not rely on that key for
stable upgrades or production distribution. Licensed under MIT.
