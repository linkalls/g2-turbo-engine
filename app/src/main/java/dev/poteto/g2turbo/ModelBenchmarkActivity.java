package dev.poteto.g2turbo;

import android.app.Activity;
import android.content.ClipData;
import android.content.ClipboardManager;
import android.content.Context;
import android.content.Intent;
import android.database.Cursor;
import android.net.Uri;
import android.os.Build;
import android.os.Bundle;
import android.provider.OpenableColumns;
import android.text.InputType;
import android.view.View;
import android.widget.Button;
import android.widget.EditText;
import android.widget.LinearLayout;
import android.widget.ScrollView;
import android.widget.TextView;

import org.json.JSONArray;
import org.json.JSONObject;

import java.io.BufferedInputStream;
import java.io.BufferedOutputStream;
import java.io.File;
import java.io.FileInputStream;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.InputStream;
import java.net.HttpURLConnection;
import java.net.URL;
import java.nio.charset.StandardCharsets;
import java.time.Instant;
import java.util.Locale;

/**
 * Real on-device GGUF benchmark: no mock model or fake tokens/sec.
 *
 * Users can choose an existing .gguf via Android SAF or explicitly download
 * the official PFN PLaMo 3 Translate 0.8B Q4_0 GGUF (~548 MB).
 * The model stays in this app's PRIVATE storage and is reused across runs.
 *
 * The JNI runner measures llama.cpp prompt prefill and autoregressive decode
 * separately for 1, 2, and 4 CPU threads.
 */
public final class ModelBenchmarkActivity extends Activity {
    static {
        System.loadLibrary("g2turbo");
    }

    private static final int PICK_GGUF_REQUEST = 501;
    private static final String OFFICIAL_Q40_URL =
            "https://huggingface.co/pfnet/plamo-3-translate-0.8b-GGUF/resolve/main/"
            + "plamo-3-translate-0.8b-Q4_0.gguf?download=true";

    private native String nativeRunGGUF(
            String modelPath, String prompt, int outputTokens);

    private Button chooseButton;
    private Button downloadButton;
    private Button runButton;
    private Button copyButton;
    private EditText inputText;
    private EditText maxTokensText;
    private TextView modelStatus;
    private TextView progress;
    private TextView result;
    private String lastJson = "";
    private boolean running = false;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);

        ScrollView scroll = new ScrollView(this);
        LinearLayout root = new LinearLayout(this);
        root.setOrientation(LinearLayout.VERTICAL);
        final int p = Math.round(18f * getResources().getDisplayMetrics().density);
        root.setPadding(p, p, p, p);

        TextView header = new TextView(this);
        header.setText("🚀 実GGUF推論の速度測定");
        header.setTextSize(22f);
        root.addView(header);

        // The real GGUF benchmark is now the launcher activity.
        // A visible navigation button keeps the older Q4_0 microbench usable.
        Button kernelBenchButton = new Button(this);
        kernelBenchButton.setText("従来のQ4_0 CPU/GPUカーネルベンチを開く");
        kernelBenchButton.setOnClickListener(v -> startActivity(
                new Intent(this, MainActivity.class)));
        root.addView(kernelBenchButton);

        TextView help = new TextView(this);
        help.setText(
                "PLaMo 3 Translate 0.8B (Q4_0) / Pixel 7 Pro\n\n"
                + "公式モデルを取得するか、端末内のGGUFを選んでね。"
                + "モデルはAPKに同梱せず、このアプリ専用領域に保存する。\n"
                + "推論は端末内で完結。計測結果は自動送信しない。\n\n"
                + "1/2/4スレッドで同じ翻訳プロンプトを処理し、"
                + "モデル読込・Prefill・Decodeをそれぞれ測定。"
                + "このモードはllama.cppのCPUバックエンドで動く"
                + "（独自G2カーネルはまだ未接続）。\n");
        help.setTextSize(14f);
        root.addView(help);

        downloadButton = new Button(this);
        downloadButton.setText("公式 PLaMo Q4_0 GGUFを取得（約548MB）");
        downloadButton.setOnClickListener(v -> downloadOfficial());
        root.addView(downloadButton);

        chooseButton = new Button(this);
        chooseButton.setText("端末内のGGUFを選ぶ");
        chooseButton.setOnClickListener(v -> openPicker());
        root.addView(chooseButton);

        modelStatus = new TextView(this);
        modelStatus.setTextSize(14f);
        root.addView(modelStatus);

        TextView textLabel = new TextView(this);
        textLabel.setText("翻訳する英文（Prefill測定用）");
        root.addView(textLabel);

        inputText = new EditText(this);
        inputText.setMinLines(4);
        inputText.setMaxLines(10);
        inputText.setSingleLine(false);
        inputText.setText(
                "Modern smartphones can now translate text locally without "
                + "sending personal information to external servers. "
                + "On-device language models have to balance low latency, "
                + "memory bandwidth, battery consumption and output accuracy. "
                + "This experiment compares several CPU thread configurations "
                + "on a Google Tensor G2 processor, measuring both prompt "
                + "processing and autoregressive text generation. "
                + "The benchmark uses exactly the same input for every run.");
        root.addView(inputText);

        TextView tokenLabel = new TextView(this);
        tokenLabel.setText("最大出力トークン数（32〜256、初期値128）");
        root.addView(tokenLabel);
        maxTokensText = new EditText(this);
        maxTokensText.setInputType(InputType.TYPE_CLASS_NUMBER);
        maxTokensText.setText("128");
        root.addView(maxTokensText);

        runButton = new Button(this);
        runButton.setText("🔥 実GGUFで 1 / 2 / 4スレッドを一括測定");
        runButton.setOnClickListener(v -> benchmark());
        root.addView(runButton);

        copyButton = new Button(this);
        copyButton.setText("実GGUFの統合JSONをコピー");
        copyButton.setEnabled(false);
        copyButton.setOnClickListener(v -> {
            ClipboardManager cm = (ClipboardManager)
                    getSystemService(Context.CLIPBOARD_SERVICE);
            if (cm != null && !lastJson.isEmpty()) {
                cm.setPrimaryClip(
                        ClipData.newPlainText("G2 real GGUF results", lastJson));
            }
        });
        root.addView(copyButton);

        progress = new TextView(this);
        progress.setTextSize(15f);
        progress.setText("準備完了。モデルを選択してね。");
        root.addView(progress);

        result = new TextView(this);
        result.setTextSize(13f);
        result.setTextIsSelectable(true);
        root.addView(result);

        scroll.addView(root);
        setContentView(scroll);
        updateButtons();
    }

    private File modelDirectory() {
        File dir = new File(getFilesDir(), "models");
        if (!dir.isDirectory() && !dir.mkdirs())
            throw new IllegalStateException("Could not create model directory");
        return dir;
    }

    private File modelFile() {
        return new File(modelDirectory(), "selected.gguf");
    }

    private boolean hasModel() {
        File f = modelFile();
        return f.isFile() && f.length() > 1024L * 1024L;
    }

    private void updateButtons() {
        boolean available = hasModel();
        chooseButton.setEnabled(!running);
        downloadButton.setEnabled(!running);
        runButton.setEnabled(!running && available);
        copyButton.setEnabled(!running && !lastJson.isEmpty());
        modelStatus.setText(available
                ? String.format(Locale.US,
                    "選択中のGGUF: %.1f MB（アプリ内に保存済み）",
                    modelFile().length() / 1000000.0)
                : "GGUFはまだ保存されていない。");
    }

    private void status(String value) {
        runOnUiThread(() -> progress.setText(value));
    }

    private void setWorking(boolean value) {
        running = value;
        updateButtons();
    }

    private void openPicker() {
        Intent intent = new Intent(Intent.ACTION_OPEN_DOCUMENT);
        intent.addCategory(Intent.CATEGORY_OPENABLE);
        intent.setType("*/*");
        startActivityForResult(intent, PICK_GGUF_REQUEST);
    }

    @Override
    protected void onActivityResult(int requestCode, int resultCode, Intent data) {
        super.onActivityResult(requestCode, resultCode, data);
        if (requestCode != PICK_GGUF_REQUEST || resultCode != RESULT_OK
                || data == null || data.getData() == null) {
            return;
        }
        final Uri uri = data.getData();
        String fileName = "";
        try (Cursor cursor = getContentResolver().query(
                uri, new String[]{OpenableColumns.DISPLAY_NAME},
                null, null, null)) {
            if (cursor != null && cursor.moveToFirst())
                fileName = cursor.getString(0);
        } catch (Exception ignored) {
            // Some document providers do not expose display names.
        }
        if (!fileName.toLowerCase(Locale.ROOT).endsWith(".gguf")) {
            status("GGUF（.gguf）ファイルを選択してね。");
            return;
        }
        setWorking(true);
        status("選択したGGUFをアプリ内へコピー中…");
        new Thread(() -> {
            File temp = new File(modelDirectory(), "selected.gguf.part");
            try (InputStream source = getContentResolver().openInputStream(uri)) {
                if (source == null) throw new IOException("Cannot open model document");
                copyAndValidate(source, temp, -1, false);
                installModel(temp);
                status("GGUFを取り込んだ。実モデル測定が可能！");
            } catch (Exception error) {
                temp.delete();
                status("取込失敗: " + error.getMessage());
            } finally {
                runOnUiThread(() -> setWorking(false));
            }
        }, "gguf-import").start();
    }

    private void downloadOfficial() {
        setWorking(true);
        status("公式PLaMo GGUFのダウンロードを開始…");
        new Thread(() -> {
            HttpURLConnection connection = null;
            File temporary = new File(modelDirectory(), "selected.gguf.part");
            try {
                URL url = new URL(OFFICIAL_Q40_URL);
                connection = (HttpURLConnection) url.openConnection();
                connection.setConnectTimeout(20000);
                connection.setReadTimeout(45000);
                connection.setInstanceFollowRedirects(true);
                connection.setRequestProperty("User-Agent", "G2-Turbo-Engine/0.5");
                final int code = connection.getResponseCode();
                if (code != HttpURLConnection.HTTP_OK)
                    throw new IOException("HTTP " + code);
                final long length = connection.getContentLengthLong();
                try (InputStream source = connection.getInputStream()) {
                    copyAndValidate(source, temporary, length, true);
                }
                installModel(temporary);
                status("公式GGUFの取得完了。Prefill/Decodeを計測できるよ！");
            } catch (Exception error) {
                temporary.delete();
                status("ダウンロード失敗: " + error.getMessage()
                        + "。端末内のGGUFを選ぶ方法でもOK。");
            } finally {
                if (connection != null) connection.disconnect();
                runOnUiThread(() -> setWorking(false));
            }
        }, "gguf-hf-download").start();
    }

    private void copyAndValidate(
            InputStream source, File destination, long total, boolean network)
            throws IOException {
        long bytes = 0L;
        long previousUpdate = 0L;
        try (BufferedInputStream in = new BufferedInputStream(source, 256 * 1024);
             BufferedOutputStream out = new BufferedOutputStream(
                     new FileOutputStream(destination), 256 * 1024)) {
            byte[] buffer = new byte[128 * 1024];
            while (true) {
                int count = in.read(buffer);
                if (count == -1) break;
                out.write(buffer, 0, count);
                bytes += count;
                long now = System.nanoTime();
                if (now - previousUpdate > 250_000_000L) {
                    previousUpdate = now;
                    String size = String.format(
                            Locale.US, "%.1fMB", bytes / 1e6);
                    String percentage = total > 0
                            ? String.format(Locale.US, " / %.1fMB (%.1f%%)",
                                    total / 1e6, 100.0 * bytes / total)
                            : "";
                    status((network ? "公式GGUFダウンロード: "
                                     : "GGUFを取り込み中: ")
                            + size + percentage);
                }
            }
            out.flush();
        }
        if (total > 0 && bytes != total)
            throw new IOException("Download truncated: " + bytes + "/" + total);
        if (bytes < 1024L * 1024L)
            throw new IOException("GGUF file unexpectedly small");
        try (FileInputStream f = new FileInputStream(destination)) {
            byte[] magic = new byte[4];
            if (f.read(magic) != 4
                    || magic[0] != 'G' || magic[1] != 'G'
                    || magic[2] != 'U' || magic[3] != 'F')
                throw new IOException("Not a valid GGUF header");
        }
    }

    private void installModel(File temporary) throws IOException {
        File dest = modelFile();
        // Both files live under the same internal FS. Atomic rename on Linux
        // ensures an interrupted import never corrupts the previous model.
        if (!temporary.renameTo(dest))
            throw new IOException("Unable to install GGUF into model directory");
    }

    /** Called by the real llama.cpp JNI computation thread. */
    public void onNativeProgress(String message) {
        status(message);
    }

    private void benchmark() {
        if (!hasModel()) {
            status("先にGGUFを選択してね。");
            return;
        }
        final String text = inputText.getText().toString().trim();
        if (text.isEmpty()) {
            status("翻訳する英文を入力してね。");
            return;
        }
        int maxOutput;
        try {
            maxOutput = Integer.parseInt(maxTokensText.getText().toString().trim());
        } catch (Exception error) {
            status("最大出力トークン数は数字で指定してね。");
            return;
        }
        if (maxOutput < 32 || maxOutput > 256) {
            status("最大出力トークン数は32〜256にしてね。");
            return;
        }

        // Official PFML3 translation prompt: en -> ja.
        // Keep explicit input/output language tags. The llama.cpp tokenizer
        // inserts the model's BOS special token.
        final String promptText =
                "<|plamo:tag|>dataset<|plamo:msg|>translation"
                + "<|plamo:tag|>input"
                + "<|plamo:key|>lang<|plamo:val|>en"
                + "<|plamo:msg|>" + text
                + "<|plamo:tag|>output"
                + "<|plamo:key|>lang<|plamo:val|>ja"
                + "<|plamo:msg|>";

        setWorking(true);
        lastJson = "";
        result.setText("");
        status("llama.cppで実GGUFをロードするよ…");

        new Thread(() -> {
            try {
                String raw = nativeRunGGUF(
                        modelFile().getAbsolutePath(), promptText, maxOutput);
                JSONObject benchmark = new JSONObject(raw);
                JSONObject root = new JSONObject();
                root.put("schema_version", 1);
                root.put("device", Build.MODEL);
                root.put("android_sdk", Build.VERSION.SDK_INT);
                root.put("date_utc", Instant.now().toString());
                root.put("model_file_bytes", modelFile().length());
                root.put("prompt_format", "PFML3 en->ja");
                root.put("native_benchmark", benchmark);

                final String formatted = root.toString(2);
                runOnUiThread(() -> {
                    lastJson = formatted;
                    result.setText(makeSummary(benchmark) + "\n\n"
                            + "===== コピー可能な統合JSON =====\n"
                            + formatted);
                    progress.setText(benchmark.optString("status").equals("error")
                            ? "計測失敗（下のJSONに理由あり）"
                            : "✅ 実GGUFの速度測定が終了。結果をコピーできるよ。");
                });
            } catch (Throwable error) {
                status("計測失敗: " + error.getMessage());
            } finally {
                runOnUiThread(() -> setWorking(false));
            }
        }, "gguf-llama-bench").start();
    }

    private static String makeSummary(JSONObject benchmark) {
        StringBuilder report = new StringBuilder();
        report.append("===== 実GGUFモデル推論結果 =====\n");
        if (benchmark.has("message")) {
            return report.append("ERROR: ")
                    .append(benchmark.optString("message")).toString();
        }
        report.append("モデル: ").append(benchmark.optString("model_description"))
              .append("\n");
        report.append(String.format(Locale.US, "モデルロード: %.1f ms\n",
                benchmark.optDouble("model_load_ms")));
        JSONArray runs = benchmark.optJSONArray("runs");
        if (runs == null) return report.toString();
        for (int i=0; i<runs.length(); i++) {
            JSONObject run = runs.optJSONObject(i);
            if (run == null) continue;
            report.append("\nCPU ").append(run.optInt("threads"))
                  .append("スレッド\n");
            if (!"ok".equals(run.optString("status"))) {
                report.append("ERROR: ").append(run.optString("message")).append("\n");
                continue;
            }
            report.append(String.format(Locale.US,
                    "Prefill: %.2f tok/s (%d tokens, %.1f ms)\n",
                    run.optDouble("prefill_tokens_per_s"),
                    run.optInt("prompt_tokens"),
                    run.optDouble("prefill_eval_ms")));
            report.append(String.format(Locale.US,
                    "Decode: %.2f tok/s (%d eval, %.1f ms)\n",
                    run.optDouble("decode_tokens_per_s"),
                    run.optInt("decode_tokens_evaluated"),
                    run.optDouble("decode_eval_ms")));
            report.append("翻訳: ").append(run.optString("generated_text")).append("\n");
        }
        return report.toString();
    }
}
