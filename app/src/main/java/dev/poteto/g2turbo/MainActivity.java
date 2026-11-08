package dev.poteto.g2turbo;

import android.app.Activity;
import android.content.ClipData;
import android.content.ClipboardManager;
import android.content.Context;
import android.content.Intent;
import android.content.res.AssetManager;
import android.os.Build;
import android.os.Bundle;
import android.view.View;
import android.widget.Button;
import android.widget.LinearLayout;
import android.widget.ScrollView;
import android.widget.TextView;

import org.json.JSONArray;
import org.json.JSONObject;

import java.time.Instant;
import java.util.ArrayList;
import java.util.List;
import java.util.Locale;

/**
 * G2 Turbo Engine v0.4 — CPU pool, affinity and KleidiAI comparison.
 *
 * 全5サイズを自動で計測してひとつのJSONへ集約する。
 * 各サイズではスカラー・ARM SDOT(1/2/4スレッド)・Vulkan 3実装を比較。
 * 途中に失敗したサイズがあっても残りは続行し、エラーもJSONに保存する。
 *
 * このアプリはGGUFを読み込まない。LLMのtoken/sベンチではない。
 */
public final class MainActivity extends Activity {
    static {
        System.loadLibrary("g2turbo");
    }

    private native String nativeBenchmark(
            int rows, int cols, int repetitions, AssetManager assets);

    private static final class Case {
        final String id;
        final String title;
        final int rows;
        final int cols;
        final int repetitions;

        Case(String id, String title, int rows, int cols, int repetitions) {
            this.id = id;
            this.title = title;
            this.rows = rows;
            this.cols = cols;
            this.repetitions = repetitions;
        }
    }

    // Fixed ordering makes measurements comparable between test runs.
    private static final Case[] CASES = {
        new Case("legacy", "従来型 768×1536", 768, 1536, 12),
        new Case("hidden", "Hidden 1536×1536", 1536, 1536, 10),
        new Case("mlp_up", "MLP Up 6144×1536", 6144, 1536, 6),
        new Case("mlp_down", "MLP Down 1536×6144", 1536, 6144, 6),
        new Case("attention_q", "Attention Q 2048×1536", 2048, 1536, 8)
    };

    private final List<Button> runButtons = new ArrayList<>();
    private Button copyButton;
    private Button shareButton;
    private TextView statusView;
    private TextView resultView;
    private String lastJson = "";

    @Override
    protected void onCreate(Bundle state) {
        super.onCreate(state);

        ScrollView scroll = new ScrollView(this);
        LinearLayout root = new LinearLayout(this);
        root.setOrientation(LinearLayout.VERTICAL);
        int padding = (int) (18 * getResources().getDisplayMetrics().density);
        root.setPadding(padding, padding, padding, padding);

        TextView heading = new TextView(this);
        heading.setText("G2 Turbo Engine v0.5.1 · カーネルベンチ");
        heading.setTextSize(23f);
        root.addView(heading);

        Button realModelButton = new Button(this);
        realModelButton.setText("🚀 実GGUFモデルのPrefill / Decode速度測定");
        realModelButton.setOnClickListener(v -> startActivity(
                new Intent(this, ModelBenchmarkActivity.class)));
        root.addView(realModelButton);

        TextView description = new TextView(this);
        description.setText(
                "Pixel 7 Pro / Tensor G2 — 1回で全ベンチ結果を収集！\n\n" +
                "5サイズ × CPU(スカラー・DOTPROD・永続プール・affinity・KleidiAI)" +
                " × Vulkan GPU(従来64・Tiled32・Subgroup)。\n" +
                "Vulkanの純演算時間はGPU Timestampで計測できる場合に記録。" +
                "対応外のGPU方式はエラー理由も残す。\n\n" +
                "※ Q4_0のマイクロベンチで、実モデルのtoken/s計測ではない。\n");
        description.setTextSize(14f);
        root.addView(description);

        Button runAll = new Button(this);
        runAll.setText("🔥 全5ケースを一括測定");
        root.addView(runAll);
        runButtons.add(runAll);
        runAll.setOnClickListener(v -> startBenchmark(true, null));

        TextView section = new TextView(this);
        section.setText("個別測定（必要な場合のみ）");
        section.setTextSize(13f);
        root.addView(section);

        for (Case item : CASES) {
            Button button = new Button(this);
            button.setText(item.title);
            root.addView(button);
            runButtons.add(button);
            button.setOnClickListener(v -> startBenchmark(false, item));
        }

        copyButton = new Button(this);
        copyButton.setText("統合JSONをコピー");
        copyButton.setEnabled(false);
        copyButton.setOnClickListener(v -> {
            if (lastJson.isEmpty()) return;
            ClipboardManager manager = (ClipboardManager)
                    getSystemService(Context.CLIPBOARD_SERVICE);
            if (manager != null) {
                manager.setPrimaryClip(
                        ClipData.newPlainText("G2 Turbo all benchmarks", lastJson));
            }
        });
        root.addView(copyButton);

        shareButton = new Button(this);
        shareButton.setText("統合JSONを共有");
        shareButton.setEnabled(false);
        shareButton.setOnClickListener(v -> {
            if (lastJson.isEmpty()) return;
            Intent send = new Intent(Intent.ACTION_SEND);
            send.setType("application/json");
            send.putExtra(Intent.EXTRA_TEXT, lastJson);
            startActivity(Intent.createChooser(send, "ベンチマーク結果を共有"));
        });
        root.addView(shareButton);

        statusView = new TextView(this);
        statusView.setTextSize(15f);
        statusView.setText("未測定。上の「全5ケースを一括測定」を押してね。");
        root.addView(statusView);

        resultView = new TextView(this);
        resultView.setTextSize(13f);
        resultView.setTextIsSelectable(true);
        resultView.setText("計測が終わると、ここに結果の概要と統合JSONが出る。");
        root.addView(resultView);

        scroll.addView(root);
        setContentView(scroll);
    }

    private void setWorking(boolean working) {
        for (Button b : runButtons) b.setEnabled(!working);
        copyButton.setEnabled(!working && !lastJson.isEmpty());
        shareButton.setEnabled(!working && !lastJson.isEmpty());
    }

    private void startBenchmark(boolean all, Case selected) {
        if (!all && selected == null) return;
        lastJson = "";
        setWorking(true);

        final Case[] runs = all ? CASES : new Case[]{selected};
        final String startedAt = Instant.now().toString();
        final long startedNs = System.nanoTime();
        statusView.setText(String.format(Locale.US, "0/%d ケース完了", runs.length));
        resultView.setText("Q4_0行列の生成とCPU/GPU計測を開始…\n" +
                "一部のGPUで実行に失敗しても、ほかのサイズは続行する。");

        new Thread(() -> {
            JSONArray output = new JSONArray();
            int succeeded = 0;
            int failed = 0;
            for (int index = 0; index < runs.length; index++) {
                final Case item = runs[index];
                final int currentIndex = index;
                runOnUiThread(() -> statusView.setText(String.format(
                        Locale.US, "%d/%d 完了 — 実行中: %s",
                        currentIndex, runs.length, item.title)));

                JSONObject entry = new JSONObject();
                try {
                    entry.put("case_id", item.id);
                    entry.put("label", item.title);
                    entry.put("rows", item.rows);
                    entry.put("cols", item.cols);
                    entry.put("repetitions", item.repetitions);

                    // A completely separate native invocation per case.
                    // Vulkan failure is recorded in nested JSON rather than
                    // aborting the following test sizes.
                    final long caseStart = System.nanoTime();
                    String raw = nativeBenchmark(
                            item.rows, item.cols, item.repetitions, getAssets());
                    entry.put("duration_ms", (System.nanoTime() - caseStart) / 1e6);
                    JSONObject data = new JSONObject(raw);
                    entry.put("measurements", data);
                    boolean cpuPass = data.optBoolean("pass", false);
                    JSONObject vk = data.optJSONObject("vulkan");
                    boolean gpuPass = vk != null && "ok".equals(vk.optString("status"));
                    entry.put("cpu_pass", cpuPass);
                    entry.put("gpu_has_passing_variant", gpuPass);
                    entry.put("status", cpuPass ? "ok" : "error");
                    if (cpuPass) succeeded++;
                    else failed++;
                } catch (Throwable error) {
                    // Capture even parse/JNI/runtime errors with accurate per-case metadata.
                    try {
                        entry.put("status", "error");
                        entry.put("error", error.getClass().getSimpleName()
                                + ": " + String.valueOf(error.getMessage()));
                    } catch (Exception ignored) {
                        // JSONObject.put on string and positive dimensions is safe.
                    }
                    failed++;
                }
                output.put(entry);

                final int done = index + 1;
                final int good = succeeded;
                final int bad = failed;
                final String summary = makeSummary(output);
                runOnUiThread(() -> {
                    statusView.setText(String.format(
                            Locale.US, "%d/%d 完了（CPU PASS: %d、失敗: %d）",
                            done, runs.length, good, bad));
                    resultView.setText(summary);
                });
            }

            JSONObject aggregate = new JSONObject();
            try {
                aggregate.put("schema_version", 2);
                aggregate.put("engine", "G2 Turbo Engine v0.4 Q4_0 GEMV");
                aggregate.put("full_llm_inference", false);
                aggregate.put("kleidiai_integrated", true);
                aggregate.put("batch_mode", all);
                aggregate.put("started_at_utc", startedAt);
                aggregate.put("finished_at_utc", Instant.now().toString());
                aggregate.put("duration_ms", (System.nanoTime() - startedNs) / 1e6);
                aggregate.put("device_manufacturer", Build.MANUFACTURER);
                aggregate.put("device_model", Build.MODEL);
                aggregate.put("android_sdk", Build.VERSION.SDK_INT);
                aggregate.put("cases_total", runs.length);
                aggregate.put("cases_cpu_pass", succeeded);
                aggregate.put("cases_error", failed);
                aggregate.put("cases", output);
                // Pretty printing is for easy human inspection and copy/paste.
                final String json = aggregate.toString(2);
                final String report = makeSummary(output);
                runOnUiThread(() -> {
                    lastJson = json;
                    setWorking(false);
                    statusView.setText(String.format(Locale.US,
                            "✅ 完了: 全%dケース中 %d成功・%d失敗。" +
                            " 統合JSONのコピー/共有が可能。",
                            runs.length, goodCount(output), runs.length - goodCount(output)));
                    resultView.setText(report +
                            "\n\n===== 全ケース統合JSON =====\n" + json);
                });
            } catch (Exception e) {
                runOnUiThread(() -> {
                    setWorking(false);
                    statusView.setText("JSONの集約に失敗: " + e.getMessage());
                    resultView.setText(makeSummary(output));
                });
            }
        }, "g2-turbo-batch-benchmark").start();
    }

    private static int goodCount(JSONArray entries) {
        int count = 0;
        for (int i = 0; i < entries.length(); i++) {
            JSONObject item = entries.optJSONObject(i);
            if (item != null && "ok".equals(item.optString("status"))) count++;
        }
        return count;
    }

    private static String makeSummary(JSONArray entries) {
        StringBuilder sb = new StringBuilder("===== 測定結果 =====\n");
        for (int i = 0; i < entries.length(); i++) {
            JSONObject item = entries.optJSONObject(i);
            if (item == null) continue;
            sb.append("\n").append(i + 1).append(". ")
                    .append(item.optString("label")).append("\n");
            JSONObject data = item.optJSONObject("measurements");
            if (data == null) {
                sb.append("ERROR: ").append(item.optString("error")).append("\n");
                continue;
            }
            sb.append(String.format(Locale.US,
                    "CPU scalar %.4fms | DOTPROD %.4fms (+量子化)\n",
                    data.optDouble("reference_ms"),
                    data.optDouble("dotprod_ms") +
                            data.optDouble("activation_quantization_ms")));
            JSONArray cpu = data.optJSONArray("cpu_thread_benchmarks");
            if (cpu != null) {
                for (int j = 0; j < cpu.length(); j++) {
                    JSONObject thread = cpu.optJSONObject(j);
                    if (thread == null) continue;
                    sb.append(String.format(Locale.US,
                            "  CPU %d threads: %.4fms\n",
                            thread.optInt("threads"),
                            thread.optDouble("total_with_activation_quant_ms")));
                }
            }
            JSONObject pool = data.optJSONObject("cpu_persistent_pool");
            if (pool != null) {
                JSONArray variants = pool.optJSONArray("variants");
                if (variants != null) {
                    for (int j=0; j<variants.length(); j++) {
                        JSONObject v = variants.optJSONObject(j);
                        if (v == null) continue;
                        sb.append(String.format(Locale.US,
                                "  Pool %s %d threads: %.4fms / pin=%s\n",
                                v.optString("strategy"),
                                v.optInt("threads"),
                                v.optDouble("total_with_activation_quant_ms"),
                                v.optBoolean("main_affinity_applied") ? "yes":"no"));
                    }
                }
            }
            JSONObject kleidi = data.optJSONObject("kleidiai");
            if (kleidi != null) {
                if (kleidi.optBoolean("pass")) {
                    sb.append(String.format(Locale.US,
                            "  KleidiAI: %.4fms + weight-pack-once %.3fms\n",
                            kleidi.optDouble("total_steady_state_ms"),
                            kleidi.optDouble("rhs_pack_once_ms")));
                } else {
                    sb.append("  KleidiAI: ").append(kleidi.optString("status"))
                        .append(" ").append(kleidi.optString("message")).append("\n");
                }
            }
            JSONObject gpu = data.optJSONObject("vulkan");
            if (gpu == null) continue;
            sb.append("GPU best: ").append(gpu.optString("best_variant"))
                    .append("\n");
            JSONArray variants = gpu.optJSONArray("variants");
            if (variants == null) continue;
            for (int j = 0; j < variants.length(); j++) {
                JSONObject v = variants.optJSONObject(j);
                if (v == null) continue;
                sb.append("  ").append(v.optString("variant")).append(": ");
                if ("ok".equals(v.optString("status"))) {
                    sb.append(String.format(Locale.US,
                            "wall %.4fms, GPU timestamp %s\n",
                            v.optDouble("hot_dispatch_wall_mean_ms"),
                            v.optBoolean("gpu_timestamp_available")
                                    ? String.format(Locale.US, "%.4fms",
                                    v.optDouble("hot_gpu_timestamp_mean_ms"))
                                    : "非対応"));
                } else {
                    sb.append(v.optString("status")).append(" ")
                            .append(v.optString("message")).append("\n");
                }
            }
        }
        return sb.toString();
    }
}
