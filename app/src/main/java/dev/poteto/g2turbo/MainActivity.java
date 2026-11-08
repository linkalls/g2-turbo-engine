package dev.poteto.g2turbo;

import android.app.Activity;
import android.content.ClipData;
import android.content.ClipboardManager;
import android.content.Context;
import android.content.res.AssetManager;
import android.os.Bundle;
import android.view.View;
import android.widget.Button;
import android.widget.LinearLayout;
import android.widget.ScrollView;
import android.widget.TextView;

import org.json.JSONObject;

import java.util.ArrayList;
import java.util.List;
import java.util.Locale;

/**
 * G2 Turbo Engine v0.2
 *
 * Pixel 7 Pro 上でQ4_0の3つの実行経路を比較する:
 * 1. スカラーFP32参照実装
 * 2. ARMv8.2 SDOT + ブロック単位INT8入力
 * 3. Vulkan GPU (Mali-G710 / 対応GPU)
 *
 * 計測は完全オフライン。実モデルの推論とは区別すること。
 * 長時間のGPU待機をUIスレッドから実行しない。
 */
public final class MainActivity extends Activity {
    static {
        System.loadLibrary("g2turbo");
    }

    private native String nativeBenchmark(
            int rows, int cols, int repetitions, AssetManager assets);

    private TextView results;
    private final List<Button> benchmarkButtons = new ArrayList<>();
    private String latestRaw = "";

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);

        ScrollView scroll = new ScrollView(this);
        LinearLayout layout = new LinearLayout(this);
        layout.setOrientation(LinearLayout.VERTICAL);
        int pad = (int) (18 * getResources().getDisplayMetrics().density);
        layout.setPadding(pad, pad, pad, pad);

        TextView heading = new TextView(this);
        heading.setText("G2 Turbo Engine v0.2");
        heading.setTextSize(23);
        layout.addView(heading);

        TextView description = new TextView(this);
        description.setText(
                "Tensor G2 / Mali-G710 — Q4_0 GEMV実験\n\n" +
                "Scalar FP32 vs ARM SDOT vs 実Vulkan GPUを比較。\n" +
                "Vulkan値はGPU実行を送信して完了待ちする時間（wall time）。" +
                "初回パイプライン生成・重み転送時間は別表示。\n\n" +
                "実際のPLaMo 3 Translate 0.8Bは hidden=1536、" +
                "intermediate=6144、16層。モデルの行列演算サイズに" +
                "近いプリセットを用意した。\n\n" +
                "※ モデル自体を動かすものではなく、tok/sは測れない。" +
                "KleidiAIはまだ未接続。GPUドライバにより非対応の場合は" +
                "失敗理由を表示する。\n");
        description.setTextSize(14);
        layout.addView(description);

        addBenchButton(layout, "従来の比較：768 × 1536", 768, 1536, 12);
        addBenchButton(layout, "Hidden型：1536 × 1536", 1536, 1536, 10);
        addBenchButton(layout, "MLP gate/up型：6144 × 1536", 6144, 1536, 6);
        addBenchButton(layout, "MLP down型：1536 × 6144", 1536, 6144, 6);
        addBenchButton(layout, "Attention Q型：2048 × 1536", 2048, 1536, 8);

        Button copyButton = new Button(this);
        copyButton.setText("最後のJSON結果をコピー");
        copyButton.setOnClickListener(v -> {
            if (latestRaw.isEmpty()) return;
            ClipboardManager manager =
                    (ClipboardManager) getSystemService(Context.CLIPBOARD_SERVICE);
            if (manager != null) {
                manager.setPrimaryClip(
                        ClipData.newPlainText("G2 Turbo benchmark", latestRaw));
            }
        });
        layout.addView(copyButton);

        results = new TextView(this);
        results.setTextSize(14);
        results.setTextIsSelectable(true);
        results.setText("上のプリセットを選ぶと比較が始まる。\n" +
                        "別途、充電状態・発熱・他アプリの動作で測定値は変わる。");
        layout.addView(results);

        scroll.addView(layout);
        setContentView(scroll);
    }

    private void addBenchButton(
            LinearLayout parent, String title, int rows, int cols, int repetitions) {
        Button button = new Button(this);
        button.setText(title);
        parent.addView(button);
        benchmarkButtons.add(button);
        button.setOnClickListener((View unused) ->
                startBenchmark(rows, cols, repetitions));
    }

    private void startBenchmark(int rows, int cols, int repetitions) {
        for (Button button : benchmarkButtons) button.setEnabled(false);
        latestRaw = "";
        results.setText(String.format(
                Locale.US,
                "%d × %d のQ4_0行列を生成・CPU測定・GPU実行中…\n" +
                "Vulkan初回はシェーダーのコンパイルに時間がかかる場合がある。",
                rows, cols));

        new Thread(() -> {
            String raw;
            try {
                raw = nativeBenchmark(rows, cols, repetitions, getAssets());
            } catch (Throwable error) {
                raw = "JNIエラー: " + error.getClass().getSimpleName() +
                        ": " + error.getMessage();
            }
            final String data = raw;
            runOnUiThread(() -> {
                latestRaw = data;
                results.setText(makeReport(data));
                for (Button button : benchmarkButtons) button.setEnabled(true);
            });
        }, "g2-turbo-vulkan-bench").start();
    }

    private String makeReport(String json) {
        try {
            JSONObject obj = new JSONObject(json);
            JSONObject vk = obj.optJSONObject("vulkan");

            final double scalarMs = obj.getDouble("reference_ms");
            final double cpuMs = obj.getDouble("dotprod_ms") +
                    obj.getDouble("activation_quantization_ms");

            StringBuilder summary = new StringBuilder();
            summary.append("===== Pixel CPU =====\n");
            summary.append(String.format(Locale.US,
                    "Scalar FP32: %.4f ms\n", scalarMs));
            summary.append(String.format(Locale.US,
                    "ARM SDOT + 入力量子化: %.4f ms\n", cpuMs));
            summary.append(String.format(Locale.US,
                    "CPU基準に対する高速化: %.2f倍\n",
                    scalarMs / Math.max(cpuMs, 1e-9)));
            summary.append(String.format(Locale.US,
                    "CPU NRMSE: %.5f / PASS: %s\n\n",
                    obj.optDouble("normalized_rmse"),
                    obj.optBoolean("pass")));

            summary.append("===== Vulkan GPU =====\n");
            if (vk != null) {
                final String status = vk.optString("status", "unknown");
                summary.append("ステータス: ").append(status).append("\n");
                summary.append("GPU: ").append(vk.optString("device")).append("\n");
                if ("ok".equals(status)) {
                    final double gpuMs =
                            vk.getDouble("hot_dispatch_wall_mean_ms");
                    summary.append(String.format(Locale.US,
                            "Warm dispatch+fence: %.4f ms\n", gpuMs));
                    summary.append(String.format(Locale.US,
                            "CPU SDOT / GPU時間比: %.2f倍\n",
                            cpuMs / Math.max(gpuMs, 1e-9)));
                    summary.append(String.format(Locale.US,
                            "GPU初期化+転送: %.2f ms\n",
                            vk.getDouble("setup_including_upload_ms")));
                    summary.append(String.format(Locale.US,
                            "GPU NRMSE: %.6f / PASS: %s\n",
                            vk.getDouble("nrmse"), vk.optBoolean("pass")));
                } else {
                    summary.append("理由: ").append(
                            vk.optString("message", "精度確認エラー")).append("\n");
                    if (vk.has("nrmse")) {
                        summary.append("GPU NRMSE: ")
                                .append(vk.getDouble("nrmse")).append("\n");
                    }
                }
            } else {
                summary.append("GPUデータなし\n");
            }

            summary.append("\n===== 生のJSON（コピー可能） =====\n");
            summary.append(json);
            return summary.toString();
        } catch (Exception ex) {
            return "結果の解析に失敗: " + ex.getMessage() + "\n\n" + json;
        }
    }
}
