package dev.poteto.g2turbo;

import android.app.Activity;
import android.os.Bundle;
import android.view.View;
import android.widget.Button;
import android.widget.LinearLayout;
import android.widget.ScrollView;
import android.widget.TextView;

/**
 * Tensor G2カーネル実験用の独立APK。
 * モデル重みやネットワーク権限は持たず、Q4_0 GEMVだけを計測する。
 */
public final class MainActivity extends Activity {
    static {
        System.loadLibrary("g2turbo");
    }

    private native String nativeBenchmark(int rows, int cols, int iterations);

    private TextView results;
    private Button startButton;

    @Override
    protected void onCreate(Bundle state) {
        super.onCreate(state);
        ScrollView scroll = new ScrollView(this);
        LinearLayout layout = new LinearLayout(this);
        layout.setOrientation(LinearLayout.VERTICAL);
        int pad = (int) (20 * getResources().getDisplayMetrics().density);
        layout.setPadding(pad, pad, pad, pad);

        TextView heading = new TextView(this);
        heading.setText("G2 Turbo Engine — Q4_0 Bench");
        heading.setTextSize(22);
        layout.addView(heading);

        TextView information = new TextView(this);
        information.setText(
                "Pixel 7 Pro / Tensor G2 向けのARM DOTPROD実験。\n\n" +
                "Q4_0重みのGEMVをCPUで比較する。\n" +
                "GPU用Vulkanシェーダーは開発中で、このAPKからは実行しない。\n" +
                "PLaMoの翻訳速度やtok/sではないので注意。\n" +
                "測定中は端末が熱くなることがある。");
        information.setTextSize(15);
        layout.addView(information);

        startButton = new Button(this);
        startButton.setText("768 × 1536 のGEMVを測定");
        layout.addView(startButton);

        results = new TextView(this);
        results.setTextSize(14);
        results.setTextIsSelectable(true);
        results.setText("測定ボタンを押すと基準演算とDOTPRODの時間・誤差をJSONで表示。\n測定結果は外部送信しない。");
        layout.addView(results);

        scroll.addView(layout);
        setContentView(scroll);

        startButton.setOnClickListener((View unused) -> {
            startButton.setEnabled(false);
            results.setText("Q4_0行列を生成して測定中…");
            new Thread(() -> {
                // UIスレッドでは実行しない。途中状態も利用者に表示する。
                final String json = nativeBenchmark(768, 1536, 12);
                runOnUiThread(() -> {
                    results.setText(json);
                    startButton.setEnabled(true);
                });
            }, "g2-turbo-benchmark").start();
        });
    }
}
