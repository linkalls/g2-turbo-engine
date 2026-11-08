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
import android.widget.Button;
import android.widget.CheckBox;
import android.widget.EditText;
import android.widget.LinearLayout;
import android.widget.ScrollView;
import android.widget.TextView;

import org.json.JSONArray;
import org.json.JSONObject;

import java.io.BufferedInputStream;
import java.io.BufferedOutputStream;
import java.io.ByteArrayOutputStream;
import java.io.File;
import java.io.FileInputStream;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.InputStream;
import java.io.OutputStream;
import java.net.HttpURLConnection;
import java.net.URL;
import java.nio.charset.StandardCharsets;
import java.time.Instant;
import java.util.ArrayList;
import java.util.LinkedHashSet;
import java.util.Locale;

/**
 * G2 Turbo v0.6: comprehensive real PLaMo GGUF benchmarking.
 *
 * Supports two separate quantization model slots with official downloads:
 * - Q4_0
 * - Q4_K_M
 *
 * Generates actual tokenizer-based prompt lengths from user-provided English
 * text, so "10,000 tokens" is a TOKEN target rather than a character count.
 * Stores results after every case, can be cancelled, and exports complete
 * JSON as a file or clipboard. No results are uploaded anywhere.
 */
public final class SweepBenchmarkActivity extends Activity {
    static { System.loadLibrary("g2turbo"); }

    private static final int PICK_MODEL = 601;
    private static final int EXPORT_JSON = 602;
    private static final String HF_BASE =
            "https://huggingface.co/pfnet/plamo-3-translate-0.8b-GGUF/resolve/main/";
    private static final String QUANT_Q40 = "Q4_0";
    private static final String QUANT_KM = "Q4_K_M";

    private native String nativeRunSweep(
            String path, String source,
            int[] promptTargets, int[] outputCaps, int[] profileCodes);
    private native void nativeCancelSweep();

    private final ArrayList<Button> mutableButtons = new ArrayList<>();
    private TextView modelsStatus;
    private TextView progress;
    private TextView overview;
    private EditText sourceText;
    private EditText prompts;
    private EditText outputs;
    private CheckBox includeQ40;
    private CheckBox includeKM;
    private CheckBox profileOne;
    private CheckBox profileTwo;
    private CheckBox profileFour;
    private CheckBox profileHybrid;
    private Button runButton;
    private Button cancelButton;
    private Button copyButton;
    private Button exportButton;
    private volatile boolean busy = false;
    private volatile boolean stopRequested = false;
    private volatile String activeModel = "";

    // Serialized by the benchmark worker: the native callback runs on that same
    // thread. UI only reads lastJson (volatile) and updates via runOnUiThread.
    private JSONObject report = new JSONObject();
    private JSONArray cases = new JSONArray();
    private JSONArray models = new JSONArray();
    private volatile String lastJson = "";
    private String pendingImportQuant = QUANT_Q40;
    private int succeeded = 0;
    private int failed = 0;
    private int plannedCases = 0;

    @Override
    protected void onCreate(Bundle state) {
        super.onCreate(state);
        ScrollView scroll = new ScrollView(this);
        LinearLayout root = new LinearLayout(this);
        root.setOrientation(LinearLayout.VERTICAL);
        int p = Math.round(17 * getResources().getDisplayMetrics().density);
        root.setPadding(p,p,p,p);

        TextView header = title("🚀 G2 Turbo v0.6 · 実GGUF全条件測定",22);
        root.addView(header);
        root.addView(title(
                "入力300〜10,000トークン級 × 最大出力32〜2,048 × CPUスレッド設定。"
                + "実際のPLaMoを動かしてPrefill・Decodeを測る。"
                + "10k入力や長い出力は数分以上かかる場合があるので、"
                + "必要な条件だけ選ぼう。各ケース完了後にJSONを自動保存する。"
                + "中断・途中失敗でも終わったケースの結果は残る。",14));

        Button oldBench = new Button(this);
        oldBench.setText("旧版の単一GGUF測定（1/2/4スレッド）を開く");
        oldBench.setOnClickListener(v -> startActivity(
                new Intent(this,ModelBenchmarkActivity.class)));
        root.addView(oldBench);

        modelsStatus = title("",15);
        root.addView(modelsStatus);
        Button q40download = action(root,
                "公式Q4_0を取得（約548 MB）",()->download(QUANT_Q40));
        Button kmdownload = action(root,
                "公式Q4_K_Mを取得（約573 MB）",()->download(QUANT_KM));
        Button q40choose = action(root,
                "端末のQ4_0 GGUFを取り込む",()->openPicker(QUANT_Q40));
        Button kmchoose = action(root,
                "端末のQ4_K_M GGUFを取り込む",()->openPicker(QUANT_KM));
        mutableButtons.add(q40download);
        mutableButtons.add(kmdownload);
        mutableButtons.add(q40choose);
        mutableButtons.add(kmchoose);

        root.addView(title("比較対象のモデル",16));
        includeQ40 = new CheckBox(this);
        includeQ40.setText("Q4_0を測定");
        includeQ40.setChecked(false);
        root.addView(includeQ40);
        includeKM = new CheckBox(this);
        includeKM.setText("Q4_K_Mを測定");
        includeKM.setChecked(true);
        root.addView(includeKM);

        root.addView(title("英文サンプル（必要な長さになるまで繰り返す）",16));
        sourceText = new EditText(this);
        sourceText.setMinLines(3);
        sourceText.setMaxLines(8);
        sourceText.setSingleLine(false);
        sourceText.setText(
                "Modern smartphones can translate documents entirely offline. "
                + "Large language models process their input in a prefill phase "
                + "and generate new tokens during a decode phase. "
                + "Longer documents require larger context windows and more "
                + "memory, so engineers must carefully measure performance, "
                + "response quality, power use and sustained thermal behavior. "
                + "This study compares several on-device inference settings "
                + "using the same source text for reproducibility.");
        root.addView(sourceText);

        root.addView(title("入力トークン数（カンマ区切り、最大12種類）",16));
        prompts=new EditText(this);
        prompts.setSingleLine(true);
        prompts.setText("300,500,1000,2000,5000,10000");
        root.addView(prompts);

        root.addView(title("最大出力トークン数（1〜2048）",16));
        outputs=new EditText(this);
        outputs.setSingleLine(true);
        outputs.setText("32,128");
        root.addView(outputs);

        Button quick = action(root,"軽量プリセット：300・500・1000 / 出力32・128",()->{
            prompts.setText("300,500,1000");
            outputs.setText("32,128");
            profileOne.setChecked(false);
            profileTwo.setChecked(true);
            profileFour.setChecked(false);
            profileHybrid.setChecked(true);
        });
        Button extensive = action(root,
                "長文プリセット：100〜10000 / 出力32・128・512",()->{
            prompts.setText("100,300,500,1000,2000,5000,10000");
            outputs.setText("32,128,512");
            profileOne.setChecked(false);
            profileTwo.setChecked(true);
            profileFour.setChecked(false);
            profileHybrid.setChecked(true);
        });
        mutableButtons.add(quick);
        mutableButtons.add(extensive);

        root.addView(title("CPU設定（Prefill / Decode スレッド数）",16));
        profileOne=new CheckBox(this);
        profileOne.setText("1 / 1");
        root.addView(profileOne);
        profileTwo=new CheckBox(this);
        profileTwo.setText("2 / 2（前回の全体最速）");
        profileTwo.setChecked(true);
        root.addView(profileTwo);
        profileFour=new CheckBox(this);
        profileFour.setText("4 / 4（Prefill最速候補）");
        root.addView(profileFour);
        profileHybrid=new CheckBox(this);
        profileHybrid.setText("4 / 2（混合・本命）");
        profileHybrid.setChecked(true);
        root.addView(profileHybrid);

        runButton=action(root,"🔥 選択した全条件で実GGUFを測定",this::startSweep);
        cancelButton=action(root,"■ 測定を中断（完了結果は保存）",this::cancelSweep);
        cancelButton.setEnabled(false);
        copyButton=action(root,"📋 途中結果を含むJSONをコピー",this::copyResults);
        exportButton=action(root,"💾 JSONをファイルとして保存・共有",this::exportResults);

        progress=title("準備完了。モデルと条件を選択してね。",15);
        root.addView(progress);
        overview=title("計測データはアプリ内に保存される。",13);
        overview.setTextIsSelectable(true);
        root.addView(overview);

        scroll.addView(root);
        setContentView(scroll);
        loadPreviousReport();
        updateButtons();
    }

    private TextView title(String value,int size) {
        TextView label=new TextView(this);
        label.setText(value);
        label.setTextSize(size);
        return label;
    }

    private Button action(LinearLayout root,String label,Runnable onClick) {
        Button btn=new Button(this);
        btn.setText(label);
        btn.setOnClickListener(v->onClick.run());
        root.addView(btn);
        return btn;
    }

    private File storage() {
        File dir=new File(getFilesDir(),"models");
        if (!dir.isDirectory() && !dir.mkdirs())
            throw new IllegalStateException("Could not create model directory");
        return dir;
    }

    private File preferredFile(String q) {
        return new File(storage(),q.equals(QUANT_Q40)?"q4_0.gguf":"q4_k_m.gguf");
    }

    private File fileFor(String q) {
        File preferred=preferredFile(q);
        if (preferred.isFile() && preferred.length()>1024*1024) return preferred;

        // Preserve v0.5.x users' already-downloaded model: no 573MB copy.
        // The exact quantization will also be read from GGUF metadata in C++.
        File legacy=new File(storage(),"selected.gguf");
        if (legacy.isFile() && legacy.length()>1024*1024) {
            boolean likelyKM=legacy.length()>560_000_000L;
            if ((likelyKM && q.equals(QUANT_KM)) ||
                (!likelyKM && q.equals(QUANT_Q40))) return legacy;
        }
        return preferred;
    }

    private void updateButtons() {
        for (Button b:mutableButtons) b.setEnabled(!busy);
        runButton.setEnabled(!busy);
        cancelButton.setEnabled(busy);
        copyButton.setEnabled(!lastJson.isEmpty());
        exportButton.setEnabled(!lastJson.isEmpty());
        modelsStatus.setText(String.format(Locale.US,
                "モデル保存状況: Q4_0 = %s / Q4_K_M = %s",
                human(fileFor(QUANT_Q40)),human(fileFor(QUANT_KM))));
    }

    private static String human(File file) {
        if (!file.exists()) return "未保存";
        return String.format(Locale.US,"%.1fMB",file.length()/1e6);
    }

    private void showProgress(String message) {
        runOnUiThread(()->progress.setText(message));
    }

    private void setBusy(boolean value) {
        busy=value;
        updateButtons();
    }

    private void openPicker(String quant) {
        pendingImportQuant=quant;
        Intent intent=new Intent(Intent.ACTION_OPEN_DOCUMENT);
        intent.addCategory(Intent.CATEGORY_OPENABLE);
        intent.setType("*/*");
        startActivityForResult(intent,PICK_MODEL);
    }

    @Override
    protected void onActivityResult(int requestCode,int resultCode,Intent data) {
        super.onActivityResult(requestCode,resultCode,data);
        if (resultCode!=RESULT_OK || data==null || data.getData()==null) return;
        Uri uri=data.getData();
        if (requestCode==EXPORT_JSON) {
            try (OutputStream out=getContentResolver().openOutputStream(uri)) {
                if(out==null) throw new IOException("Cannot create JSON document");
                out.write(lastJson.getBytes(StandardCharsets.UTF_8));
                out.flush();
                showProgress("JSONを書き出したよ。ファイルアプリから共有できる。");
            } catch(Exception e) {showProgress("書き出し失敗: "+e.getMessage());}
            return;
        }
        if (requestCode!=PICK_MODEL) return;
        String filename="";
        try(Cursor cursor=getContentResolver().query(
                uri,new String[]{OpenableColumns.DISPLAY_NAME},null,null,null)) {
            if(cursor!=null && cursor.moveToFirst()) filename=cursor.getString(0);
        }catch(Exception ignored){}
        if(!filename.toLowerCase(Locale.ROOT).endsWith(".gguf")) {
            showProgress("拡張子 .gguf のモデルを選んでね");
            return;
        }
        final String quant=pendingImportQuant;
        setBusy(true);
        new Thread(()->{
            File partial=new File(storage(),quant+".part");
            try(InputStream stream=getContentResolver().openInputStream(uri)){
                if(stream==null)throw new IOException("Unable to open model");
                copy(stream,partial,-1);
                commit(partial,quant);
                showProgress(quant+"を保存できた！");
            }catch(Exception e) {
                partial.delete();
                showProgress("取り込み失敗: "+e.getMessage());
            }finally {runOnUiThread(()->setBusy(false));}
        },"gguf-import").start();
    }

    private void download(String quant) {
        setBusy(true);
        new Thread(()->{
            File partial=new File(storage(),quant+".part");
            HttpURLConnection connection=null;
            try {
                String url=HF_BASE+"plamo-3-translate-0.8b-"+quant+".gguf?download=true";
                connection=(HttpURLConnection)new URL(url).openConnection();
                connection.setConnectTimeout(20000);
                connection.setReadTimeout(60000);
                connection.setInstanceFollowRedirects(true);
                connection.setRequestProperty("User-Agent","G2-Turbo/0.6");
                int http=connection.getResponseCode();
                if(http!=200)throw new IOException("HTTP "+http);
                long size=connection.getContentLengthLong();
                try(InputStream stream=connection.getInputStream()){
                    copy(stream,partial,size);
                }
                commit(partial,quant);
                showProgress("公式"+quant+"のダウンロードが完了！");
            }catch(Exception e) {
                partial.delete();
                showProgress("取得失敗: "+e.getMessage());
            }finally {
                if(connection!=null)connection.disconnect();
                runOnUiThread(()->setBusy(false));
            }
        },"gguf-download").start();
    }

    private void copy(InputStream source,File dst,long expected) throws IOException {
        long total=0,last=0;
        try(BufferedInputStream in=new BufferedInputStream(source,256*1024);
            BufferedOutputStream out=new BufferedOutputStream(
                    new FileOutputStream(dst),256*1024)){
            byte[] buffer=new byte[128*1024];
            int count;
            while((count=in.read(buffer))!=-1){
                out.write(buffer,0,count);
                total+=count;
                long now=System.nanoTime();
                if(now-last>350_000_000L) {
                    last=now;
                    final long current=total;
                    showProgress(String.format(Locale.US,
                            "GGUF取得中: %.1f MB%s",
                            current/1e6,expected>0?String.format(Locale.US,
                              " / %.1f MB",expected/1e6):""));
                }
            }
        }
        if(expected>0 && total!=expected)
            throw new IOException("Truncated download");
        if(total<1000000)throw new IOException("Model file too small");
        try(FileInputStream in=new FileInputStream(dst)){
            byte[] magic=new byte[4];
            if(in.read(magic)!=4||magic[0]!='G'||magic[1]!='G'
                    ||magic[2]!='U'||magic[3]!='F')
                throw new IOException("Not GGUF: missing GGUF magic");
        }
    }

    private void commit(File partial,String quant) throws IOException {
        File output=preferredFile(quant);
        if(!partial.renameTo(output))
            throw new IOException("Cannot install GGUF");
    }

    private int[] parseTargets(String value,int min,int max) {
        LinkedHashSet<Integer> parsed=new LinkedHashSet<>();
        for(String part:value.split(",")) {
            if(part.trim().isEmpty())continue;
            int number=Integer.parseInt(part.trim());
            if(number<min||number>max)
                throw new IllegalArgumentException("許容範囲: "+min+"〜"+max);
            parsed.add(number);
        }
        if(parsed.isEmpty()||parsed.size()>12)
            throw new IllegalArgumentException("1〜12種類のトークン数を指定してね");
        int[] ret=new int[parsed.size()];
        int i=0;for(int item:parsed)ret[i++]=item;
        return ret;
    }

    private int[] profiles() {
        ArrayList<Integer> result=new ArrayList<>();
        if(profileOne.isChecked())result.add(0);
        if(profileTwo.isChecked())result.add(1);
        if(profileFour.isChecked())result.add(2);
        if(profileHybrid.isChecked())result.add(3);
        int[] ret=new int[result.size()];
        for(int i=0;i<ret.length;++i)ret[i]=result.get(i);
        return ret;
    }

    private void resetReport() throws Exception {
        report=new JSONObject();
        cases=new JSONArray();
        models=new JSONArray();
        report.put("schema_version",2);
        report.put("app","G2 Turbo Engine v0.6 GGUF Sweep");
        report.put("device",Build.MODEL);
        report.put("android_sdk",Build.VERSION.SDK_INT);
        report.put("started_at_utc",Instant.now().toString());
        report.put("prompt_corpus","user_input_repeated_tokenizer_targeted");
        report.put("prompt_format","PFML3 en->ja");
        report.put("note","CPU llama.cpp baseline; no G2 custom kernel; "
                +"prompt targets approximate; output may end before cap");
        report.put("model_runs",models);
        report.put("cases",cases);
        succeeded=0;failed=0;
        refreshPersistedReport();
    }

    private File reportFile() {
        File folder=new File(getFilesDir(),"results");
        if(!folder.exists()&&!folder.mkdirs())
            throw new IllegalStateException("Cannot create results directory");
        return new File(folder,"latest-sweep.json");
    }

    private void refreshPersistedReport() throws Exception {
        report.put("cases_completed",cases.length());
        report.put("cases_ok",succeeded);
        report.put("cases_error",failed);
        lastJson=report.toString(2);
        File dest=reportFile();
        File partial=new File(dest.getParentFile(),"latest-sweep.json.part");
        try(FileOutputStream out=new FileOutputStream(partial)){
            out.write(lastJson.getBytes(StandardCharsets.UTF_8));
            out.getFD().sync();
        }
        if(!partial.renameTo(dest))
            throw new IOException("Cannot save interim report");
        final String compact=summary();
        runOnUiThread(()->{
            overview.setText(compact);
            copyButton.setEnabled(true);
            exportButton.setEnabled(true);
        });
    }

    private void loadPreviousReport() {
        File file=reportFile();
        if(!file.isFile())return;
        try(FileInputStream in=new FileInputStream(file)){
            if (file.length()>20_000_000L) throw new IOException("Report exceeds 20MB");
            ByteArrayOutputStream buffer=new ByteArrayOutputStream(
                    (int)Math.min(file.length(),1_000_000L));
            byte[] chunk=new byte[16384];
            int read;
            while((read=in.read(chunk))!=-1)buffer.write(chunk,0,read);
            report=new JSONObject(buffer.toString(StandardCharsets.UTF_8.name()));
            cases=report.optJSONArray("cases");
            models=report.optJSONArray("model_runs");
            if(cases==null)cases=new JSONArray();
            if(models==null)models=new JSONArray();
            succeeded=report.optInt("cases_ok");
            failed=report.optInt("cases_error");
            lastJson=report.toString(2);
            overview.setText(summary()+"\n前回の途中結果を復元した。");
        }catch(Exception ignored) {
            lastJson="";
        }
    }

    private String summary() {
        StringBuilder b=new StringBuilder("進捗: "+cases.length()
                +"/"+plannedCases+" ケース（成功"+succeeded+"、失敗"+failed+"）\n");
        int start=Math.max(0,cases.length()-8);
        for(int i=start;i<cases.length();i++){
            JSONObject c=cases.optJSONObject(i);
            if(c==null)continue;
            b.append(c.optString("model_slot")).append(" / ")
             .append(c.optInt("actual_prompt_tokens")).append(" in / ")
             .append(c.optInt("max_output_tokens")).append(" max / ")
             .append(c.optInt("prefill_threads")).append("/")
             .append(c.optInt("decode_threads")).append(" threads: ");
            if("ok".equals(c.optString("status"))) {
                b.append(String.format(Locale.US,"%.1f prefill, %.1f decode tok/s",
                         c.optDouble("prefill_tok_s"),c.optDouble("decode_tok_s")));
                b.append(" (out ").append(c.optInt("output_tokens_generated")).append(")");
            }else b.append("error: ").append(c.optString("message"));
            b.append("\n");
        }
        b.append("\n※出力の上限到達前に翻訳が終了する場合がある。");
        return b.toString();
    }

    private void startSweep() {
        if(busy)return;
        final int[] lengths,outputLengths,modes;
        try {
            lengths=parseTargets(prompts.getText().toString(),32,12000);
            outputLengths=parseTargets(outputs.getText().toString(),1,2048);
            modes=profiles();
            if(modes.length==0)throw new IllegalArgumentException("CPU設定を選んでね");
        }catch(Exception e){
            showProgress("測定設定に問題あり: "+e.getMessage());
            return;
        }
        final ArrayList<String> slots=new ArrayList<>();
        if(includeQ40.isChecked())slots.add(QUANT_Q40);
        if(includeKM.isChecked())slots.add(QUANT_KM);
        if(slots.isEmpty()) {
            showProgress("測るGGUFモデルを選んでね。");
            return;
        }
        for(String q:slots) {
            File f=fileFor(q);
            if(!f.isFile()||f.length()<1000000) {
                showProgress(q+"のGGUFがない。先に取得・取り込みしてね。");
                return;
            }
        }
        final String source=sourceText.getText().toString().trim();
        if(source.isEmpty()) {
            showProgress("英文サンプルを入れてね");
            return;
        }
        plannedCases=slots.size()*lengths.length*outputLengths.length*modes.length;
        if(plannedCases>180) {
            showProgress("180ケースを超えた。トークン数かCPU設定を減らしてね。");
            return;
        }
        setBusy(true);
        stopRequested=false;
        new Thread(()->{
            try {
                resetReport();
                report.put("requested_prompt_tokens",new JSONArray(lengths));
                report.put("requested_max_output_tokens",new JSONArray(outputLengths));
                report.put("requested_profiles",new JSONArray(modes));
                report.put("total_cases_planned",plannedCases);
                refreshPersistedReport();
                for(String slot:slots) {
                    if(stopRequested)break;
                    activeModel=slot;
                    showProgress(slot+": モデルロード");
                    String raw=nativeRunSweep(
                            fileFor(slot).getAbsolutePath(),source,lengths,outputLengths,modes);
                    JSONObject response=new JSONObject(raw);
                    if(!"ok".equals(response.optString("status"))){
                        JSONObject record=new JSONObject();
                        record.put("model_slot",slot);
                        record.put("error",response.optString("message"));
                        record.put("status","error");
                        models.put(record);
                    }
                    report.put("last_model_result",response);
                    refreshPersistedReport();
                }
                report.put("ended_at_utc",Instant.now().toString());
                report.put("stopped_by_user",stopRequested);
                refreshPersistedReport();
                showProgress(stopRequested?"中断した。途中までの結果は保存済み。"
                        :"✅ 測定終了。JSONコピー・ファイル保存が可能！");
            }catch(Throwable error) {
                showProgress("測定エラー: "+error.getMessage()
                        +"（完了済みケースはファイルに保存）");
            }finally {
                activeModel="";
                runOnUiThread(()->setBusy(false));
            }
        },"real-gguf-sweep").start();
    }

    private void cancelSweep() {
        stopRequested=true;
        nativeCancelSweep();
        showProgress("中断要求を送信。現在の演算が終わると停止する。");
    }

    /** JNI calls this on the experiment worker thread. Save every result. */
    public void onNativeCaseResult(String json) {
        try {
            JSONObject item=new JSONObject(json);
            JSONObject data=item.optJSONObject("data");
            if(data==null)return;
            data.put("model_slot",activeModel);
            if("model_meta".equals(item.optString("kind"))) {
                models.put(data);
                // An imported file may have been assigned to the wrong slot.
                // Check GGUF metadata, not file extension, before interpreting
                // a quantization comparison as valid.
                String actual=data.optString("model_quantization").toUpperCase(Locale.ROOT);
                boolean matching=(activeModel.equals(QUANT_Q40)&&actual.contains("Q4_0"))
                        || (activeModel.equals(QUANT_KM)&&actual.contains("Q4_K"));
                if(!matching) {
                    stopRequested=true;
                    nativeCancelSweep();
                    showProgress("モデルの形式が違う: "+activeModel+" に "+actual
                            +" が入っている。処理を中止した。");
                }
            } else if("case".equals(item.optString("kind"))){
                cases.put(data);
                if("ok".equals(data.optString("status")))succeeded++;
                else failed++;
            }
            refreshPersistedReport();
        }catch(Exception error) {
            showProgress("途中結果保存エラー: "+error.getMessage());
        }
    }

    /** JNI calls this on the experiment worker thread. */
    public void onNativeProgress(String message) {
        showProgress(activeModel+" / "+message);
    }

    private void copyResults() {
        if(lastJson.isEmpty())return;
        ClipboardManager cm=(ClipboardManager)getSystemService(Context.CLIPBOARD_SERVICE);
        if(cm!=null) {
            cm.setPrimaryClip(ClipData.newPlainText("GGUF real-model sweep",lastJson));
            showProgress("統合JSONをコピーした！");
        }
    }

    private void exportResults() {
        if(lastJson.isEmpty())return;
        Intent intent=new Intent(Intent.ACTION_CREATE_DOCUMENT);
        intent.addCategory(Intent.CATEGORY_OPENABLE);
        intent.setType("application/json");
        intent.putExtra(Intent.EXTRA_TITLE,"g2-real-gguf-sweep.json");
        startActivityForResult(intent,EXPORT_JSON);
    }
}
