from io import BytesIO
from datetime import datetime
from pathlib import Path
import sqlite3
import os
import json
import pandas as pd
from flask import Flask, request, render_template_string, send_file, redirect, url_for
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

app = Flask(__name__)
FEATURES = ["pH", "TDS_ppm", "Temperature_C"]
TARGET = "Quality"
last_output = None
last_run = None

# Render: set AQUACROP_DB_PATH to a path on a mounted persistent disk,
# for example /var/data/aquacrop_history.db. Locally, use this folder.
DB_PATH = os.environ.get(
    "AQUACROP_DB_PATH",
    str(Path(__file__).resolve().parent / "aquacrop_history.db")
)

PAGE = r"""
<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>AquaCrop | Water Quality Intelligence</title>
<style>
:root{font-family:Inter,Segoe UI,Arial,sans-serif;color:#18343b;background:#f3f7f6}
*{box-sizing:border-box}body{margin:0}.topbar{background:#123f46;color:#fff;padding:18px 5vw;display:flex;align-items:center;justify-content:space-between;gap:15px}
.brand{font-size:21px;font-weight:800;letter-spacing:.2px}.brand span{color:#72dfc1}.topnote{font-size:12px;color:#c5e2df}
.hero{background:linear-gradient(120deg,#15565d,#167e76);color:#fff;padding:38px 5vw 42px}.hero-inner{max-width:1180px;margin:auto}.eyebrow{text-transform:uppercase;letter-spacing:2px;font-size:11px;font-weight:800;color:#a9f0db}.hero h1{font-size:clamp(28px,4vw,42px);margin:10px 0}.hero p{max-width:720px;line-height:1.6;color:#e1f3ef;margin:0}
main{max-width:1180px;margin:25px auto;padding:0 18px}.section-title{font-size:19px;margin:0 0 14px}.card{background:#fff;border:1px solid #e0eae8;border-radius:16px;padding:22px;margin-bottom:18px;box-shadow:0 7px 24px #173c4408}
.grid{display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:14px}.two{display:grid;grid-template-columns:1.05fr .95fr;gap:18px}
label{display:block;font-weight:700;font-size:13px;margin:0 0 8px}input[type=file]{width:100%;padding:12px;border:1px solid #cbdcd9;border-radius:9px;background:#fbfdfc}
button,.download{display:inline-block;background:#147d70;color:#fff;padding:12px 17px;border:0;border-radius:9px;font-weight:750;text-decoration:none;cursor:pointer;font-size:14px}button:hover,.download:hover{background:#0e6158}
.muted{color:#668078;font-size:13px;line-height:1.5}.warning{background:#fff7df;border:1px solid #ead59b;padding:13px 15px;border-radius:10px;color:#725619;line-height:1.5;font-size:13px}.error{background:#fff0ef;border:1px solid #e6b7b3;padding:13px;border-radius:10px;color:#8c2820}
.metric{background:#f6faf9;border:1px solid #e4eeeb;border-radius:13px;padding:17px}.metric .label{font-size:12px;color:#648078}.metric strong{display:block;font-size:25px;margin-top:8px;overflow-wrap:anywhere}.metric .sub{font-size:11px;color:#7a928b;margin-top:6px}
.table-wrap{overflow-x:auto}table{width:100%;border-collapse:collapse;font-size:13px;min-width:650px}th,td{text-align:left;padding:12px;border-bottom:1px solid #e6eeec;vertical-align:top}th{background:#f3f8f7;color:#42635c;font-size:11px;text-transform:uppercase;letter-spacing:.5px}tbody tr:hover{background:#fbfdfc}
.pill{display:inline-block;background:#e2f5ee;color:#17644e;padding:5px 9px;border-radius:20px;font-size:11px;font-weight:800}.chart-row{margin:15px 0}.chart-label{display:flex;justify-content:space-between;gap:12px;font-size:12px;margin-bottom:7px;color:#47645d}.track{height:9px;background:#e9f0ee;border-radius:20px;overflow:hidden}.bar{height:100%;background:linear-gradient(90deg,#31b89a,#147d70);border-radius:20px}.history-note{border-left:3px solid #8bd8c5;padding:3px 0 3px 12px;color:#607970;font-size:13px;line-height:1.5}
footer{text-align:center;color:#799087;padding:25px;font-size:12px}
@media(max-width:760px){.grid{grid-template-columns:1fr}.two{grid-template-columns:1fr}.topnote{display:none}.hero{padding:28px 20px}main{padding:0 12px}.card{padding:17px}}
.success{padding:12px 14px;border:1px solid #69b99d;border-radius:10px;background:#eaf8f1;color:#176b50;margin:12px 0;font-weight:600}
</style>
</head>
<body>
<div class="topbar"><div class="brand">💧 Aqua<span>Crop</span></div><div class="topnote">WATER QUALITY • AGRICULTURAL SCREENING PROTOTYPE</div></div>
<header class="hero"><div class="hero-inner"><div class="eyebrow">Water intelligence dashboard</div><h1>Understand your irrigation water.</h1><p>Review pH, total dissolved solids and temperature, then run a machine-learning screening model on labelled samples. Use verified laboratory data before making agricultural decisions.</p></div></header>
<main>
<section class="card">
<h2 class="section-title">Run a new analysis</h2>
<p class="muted">Upload a labelled training dataset and a separate CSV containing the samples you want to screen.</p>
<form method="post" enctype="multipart/form-data">
<div class="grid">
<div><label for="training">1. Training dataset (.csv)</label><input id="training" type="file" name="training" accept=".csv" required><p class="muted">Required columns: pH, TDS_ppm, Temperature_C, Quality.</p></div>
<div><label for="readings">2. New water readings (.csv)</label><input id="readings" type="file" name="readings" accept=".csv" required><p class="muted">Required columns: pH, TDS_ppm, Temperature_C. Sample_ID is optional.</p></div>
<div><label>3. Start analysis</label><p class="muted">The model is trained again each time you submit the form.</p><button type="submit">Analyse water samples →</button></div>
</div>
</form>
{% if error %}<div class="error">{{ error }}</div>{% endif %}
{% if warning %}<div class="warning" style="margin-top:14px"><b>Data-quality notice:</b> {{ warning }}</div>{% endif %}
</section>
{% if result %}
<section>
<h2 class="section-title">Analysis overview</h2>
<div class="grid">
<div class="metric"><div class="label">Training rows used</div><strong>{{ n_train }}</strong><div class="sub">Rows remaining after invalid values were removed</div></div>
<div class="metric"><div class="label">Samples screened</div><strong>{{ n_readings }}</strong><div class="sub">New readings included in this run</div></div>
<div class="metric"><div class="label">Hold-out accuracy</div><strong style="font-size:20px">{{ accuracy }}</strong><div class="sub">Only a small internal test; not proof of field performance</div></div>
</div>
</section>
<div class="two" style="margin-top:18px">
<section class="card"><h2 class="section-title">Sample measurement summary</h2>
{% for item in summaries %}
<div class="chart-row"><div class="chart-label"><span>{{ item.name }}</span><b>{{ item.value }}</b></div><div class="track"><div class="bar" style="width:{{ item.width }}%"></div></div><div class="muted">{{ item.note }}</div></div>
{% endfor %}
<p class="muted">Bars are scaled within this displayed set for visual comparison; they are not risk scores or pass/fail limits.</p>
</section>
<section class="card"><h2 class="section-title">Predicted categories</h2>
{% for item in categories %}
<div class="chart-row"><div class="chart-label"><span>{{ item.name }}</span><b>{{ item.count }}</b></div><div class="track"><div class="bar" style="width:{{ item.width }}%"></div></div></div>
{% endfor %}
{% if not categories %}<p class="muted">No predictions to display.</p>{% endif %}
<div class="history-note">Latest run: {{ run_time }}. Every analysed sample is also appended to Saved prediction history.</div>
</section></div>
<section class="card"><h2 class="section-title">Predictions and screening notes</h2><div class="table-wrap"><table><thead><tr>{% for h in headers %}<th>{{ h }}</th>{% endfor %}</tr></thead><tbody>{% for row in rows %}<tr>{% for cell in row %}<td>{{ cell }}</td>{% endfor %}</tr>{% endfor %}</tbody></table></div><p style="margin:18px 0 0"><a class="download" href="/download">Download results CSV ↓</a></p></section>
{% endif %}
<section class="card"><h2 class="section-title">Saved prediction history</h2>
{% if request.args.get('history_cleared') %}<div class="success" role="status">Prediction history cleared successfully.</div>{% endif %}
<p class="muted">Each analysed sample is saved to the local SQLite database with its run timestamp. History persists across application restarts when the database file is on persistent storage.</p>
<div class="table-wrap"><table><thead><tr><th>Run time</th><th>Sample ID</th><th>pH</th><th>TDS (ppm)</th><th>Temp (°C)</th><th>Predicted category</th></tr></thead>
<tbody>{% for h in history %}<tr><td>{{ h[0] }}</td><td>{{ h[1] }}</td><td>{{ h[2] }}</td><td>{{ h[3] }}</td><td>{{ h[4] }}</td><td>{{ h[5] }}</td></tr>{% endfor %}</tbody></table></div>
{% if not history %}<p class="muted">No saved analyses yet. Upload CSV files above to create the first history entry.</p>{% endif %}
<div style="display:flex;gap:12px;flex-wrap:wrap;align-items:center;margin-top:16px">
<a class="download" href="/history/download">Download full history CSV ↓</a>
<form method="POST" action="/history/clear" onsubmit="return confirm('Permanently delete ALL saved prediction history? Download a backup first if you need it. This will not delete your uploaded CSV files or trained model.');" style="margin:0">
<button type="submit" class="btn btn-secondary">Clear prediction history</button>
</form>
</div>
</section>
<section class="card" id="trend-analytics">
<h2 class="section-title">Measurement trends & quick indicators</h2>
<p class="muted">Charts use saved CSV-analysis samples in history, ordered from oldest to newest among the latest 100 records. These are sensor trends, not proof of irrigation suitability.</p>
<div class="grid">
<div class="metric"><div class="label">Latest saved pH</div><strong id="trend-ph">—</strong><div class="sub">pH units</div></div>
<div class="metric"><div class="label">Latest saved TDS</div><strong id="trend-tds">—</strong><div class="sub">ppm from your TDS module</div></div>
<div class="metric"><div class="label">Latest saved temperature</div><strong id="trend-temp">—</strong><div class="sub">°C</div></div>
</div>
<div class="grid" style="margin-top:14px">
<div class="metric"><div class="label">pH history</div><canvas id="chart-ph" height="150" style="width:100%"></canvas></div>
<div class="metric"><div class="label">TDS history</div><canvas id="chart-tds" height="150" style="width:100%"></canvas></div>
<div class="metric"><div class="label">Temperature history</div><canvas id="chart-temp" height="150" style="width:100%"></canvas></div>
</div>
<div id="trend-status" class="history-note" style="margin-top:14px">Run an analysis to build your trend history.</div>
<p class="muted" style="margin-top:12px">Estimated EC (electrical conductivity) is calculated as TDS ÷ 0.64 and shown in µS/cm only as a rough estimate. The conversion factor varies with dissolved-ion composition and sensor calibration. It is not an independent EC measurement. Sodium, SAR, calcium, magnesium, hardness, chloride and bicarbonate need separate laboratory tests or appropriate validated sensors.</p>
</section>

<section class="card" id="improvement-advisor">
<h2 class="section-title">Water Improvement Advisor</h2>
<p class="muted">Enter a fresh water sample to get practical next steps for pH, temperature and TDS. This is screening guidance, not a chemical-dose calculator or proof of irrigation suitability.</p>
<div style="display:grid;grid-template-columns:repeat(auto-fit,minmax(170px,1fr));gap:14px">
<div><label for="advisor-ph">pH (0–14)</label><input id="advisor-ph" type="number" min="0" max="14" step="0.01" placeholder="e.g. 7.2"></div>
<div><label for="advisor-tds">TDS (ppm)</label><input id="advisor-tds" type="number" min="0" step="1" placeholder="e.g. 450"></div>
<div><label for="advisor-temp">Temperature (°C)</label><input id="advisor-temp" type="number" min="-5" max="100" step="0.1" placeholder="e.g. 28"></div>
</div>
<div style="display:flex;gap:10px;flex-wrap:wrap;margin-top:14px">
<button type="button" class="btn btn-primary" onclick="getImprovementAdvice()">Get improvement advice</button>
<button type="button" class="btn btn-secondary" onclick="clearImprovementAdvice()">Clear</button>
</div>
<div id="advisor-results" aria-live="polite" style="margin-top:16px"></div>
</section>

<section class="card"><h2 class="section-title">Parameter improvement guide</h2>
<p class="muted">Use these steps to investigate and manage readings. The website cannot physically treat water, and a single reading is not enough to prescribe chemical treatment.</p>
<div class="table-wrap"><table style="min-width:600px"><thead><tr><th>Parameter</th><th>If the reading is outside the screening range</th><th>Practical next steps</th></tr></thead><tbody>
<tr><td><b>pH</b></td><td>Below 6.5 or above 8.4</td><td>1. Rinse and calibrate the pH probe with fresh, suitable buffer solutions. 2. Repeat the reading on a fresh sample. 3. If it remains abnormal, test alkalinity and bicarbonate in a lab. 4. Only consider pH correction after a lab test and advice from an agricultural/water specialist; do not add acid or alkali based on this reading alone.</td></tr>
<tr><td><b>TDS</b></td><td>Above 450 ppm is an elevated-screening flag here; above 2,000 ppm is a high flag</td><td>1. Confirm with an electrical-conductivity (EC) meter and a lab test. 2. Test sodium/SAR and chloride before choosing a response. 3. If suitable and available, consider blending with a verified lower-salinity source. 4. Leaching may help only when soil drainage, crop tolerance and water quality are suitable; seek agronomy advice first.</td></tr>
<tr><td><b>Temperature</b></td><td>Above 35 °C triggers a sensor/site-check reminder in this prototype</td><td>1. Check the probe against a reference thermometer. 2. Measure consistently, away from direct sunlight, and record time and sampling location. 3. If water is stored in a tank, shade/insulate it where practical and safe. Temperature alone does not tell whether irrigation water is chemically suitable.</td></tr>
</tbody></table></div>
<div class="warning" style="margin-top:14px"><b>Important:</b> these values are preliminary software screening flags, not universal crop-specific limits. For coconut or any other crop, decisions should use local soil conditions, crop stage, irrigation method and laboratory results.</div>
</section>
<section class="card"><h2 class="section-title">How to interpret AquaCrop</h2>
<div class="warning"><b>Prototype limitation:</b> pH, TDS and temperature do not directly measure sodium, SAR, chloride, bicarbonate or hardness. The model learns the labels in your training CSV, so unreliable labels produce unreliable predictions. Do not use this prototype alone to decide whether water is suitable for irrigation.</div>
<p class="muted">For a research-grade evaluation, use independent samples with laboratory-confirmed measurements and documented irrigation-water criteria. Mark simulated or illustrative records clearly and never report them as field or laboratory data. Estimated EC and screening flags are derived indicators only; they do not expand what the physical sensors can measure.</p>
</section>
</main><footer>AquaCrop • Water quality screening prototype • Not a laboratory instrument</footer>

<script>
const aquaTrendRows = {{ trend_rows|default([], true)|tojson }};
function drawTrend(canvasId, values, label) {
  const canvas=document.getElementById(canvasId); if(!canvas) return;
  const ctx=canvas.getContext('2d'); const dpr=window.devicePixelRatio||1;
  const w=Math.max(240,canvas.clientWidth||280), h=150;
  canvas.width=w*dpr; canvas.height=h*dpr; ctx.scale(dpr,dpr); ctx.clearRect(0,0,w,h);
  if(!values.length){ctx.fillStyle='#668078';ctx.font='12px sans-serif';ctx.fillText('No saved readings yet',10,70);return;}
  const pad=25, min=Math.min(...values), max=Math.max(...values), span=(max-min)||1;
  ctx.strokeStyle='#dce8e5';ctx.lineWidth=1;
  for(let i=0;i<4;i++){let y=pad+(h-2*pad)*i/3;ctx.beginPath();ctx.moveTo(pad,y);ctx.lineTo(w-pad,y);ctx.stroke();}
  ctx.strokeStyle='#147d70';ctx.lineWidth=2;ctx.beginPath();
  values.forEach((v,i)=>{const x=pad+(w-2*pad)*(values.length===1?0.5:i/(values.length-1));const y=h-pad-(v-min)/span*(h-2*pad);if(i===0)ctx.moveTo(x,y);else ctx.lineTo(x,y);});ctx.stroke();
  ctx.fillStyle='#47645d';ctx.font='11px sans-serif';ctx.fillText(max.toFixed(2),2,pad+3);ctx.fillText(min.toFixed(2),2,h-pad+3);ctx.fillText(values.length+' samples',w-75,h-5);
}
(function renderTrends(){
 const rows=aquaTrendRows.slice().reverse();
 if(!rows.length)return;
 const latest=rows[rows.length-1];
 document.getElementById('trend-ph').textContent=Number(latest[2]).toFixed(2);
 document.getElementById('trend-tds').textContent=Number(latest[3]).toFixed(1)+' ppm';
 document.getElementById('trend-temp').textContent=Number(latest[4]).toFixed(1)+' °C';
 drawTrend('chart-ph',rows.map(r=>Number(r[2])),'pH');
 drawTrend('chart-tds',rows.map(r=>Number(r[3])),'TDS');
 drawTrend('chart-temp',rows.map(r=>Number(r[4])),'Temperature');
 const ec=(Number(latest[3])/0.64).toFixed(0);
 const ph=Number(latest[2]),tds=Number(latest[3]);
 const flags=[]; if(ph<6.5)flags.push('pH below preliminary screening band'); else if(ph>8.4)flags.push('pH above preliminary screening band');
 if(tds>2000)flags.push('high TDS screening flag'); else if(tds>450)flags.push('elevated TDS screening flag');
 document.getElementById('trend-status').textContent='Latest approximate EC: '+ec+' µS/cm (TDS ÷ 0.64; rough estimate only). '+(flags.length?'Review: '+flags.join('; ')+'.':'No pH/TDS flag under this prototype’s preliminary rules.')+' These rules are not universal crop limits.';
})();
function getImprovementAdvice() {
  const phText=document.getElementById('advisor-ph').value.trim();
  const tdsText=document.getElementById('advisor-tds').value.trim();
  const tempText=document.getElementById('advisor-temp').value.trim();
  const out=document.getElementById('advisor-results');
  if (!phText || !tdsText || !tempText) { out.innerHTML='<div class="warning">Please enter all three readings.</div>'; return; }
  const ph=Number(phText), tds=Number(tdsText), temp=Number(tempText);
  if (!Number.isFinite(ph)||ph<0||ph>14||!Number.isFinite(tds)||tds<0||!Number.isFinite(temp)||temp< -5||temp>100) {
    out.innerHTML='<div class="warning">Check entries: pH must be 0–14, TDS must be non-negative, and temperature must be between −5 and 100 °C.</div>'; return;
  }
  const cards=[];
  if(ph<6.5) cards.push(['pH — low/acidic screening flag','Calibrate the probe with suitable fresh buffer solutions and repeat the reading. If it remains low, test alkalinity and full water chemistry. A specialist should determine whether correction is needed; do not add lime or alkali based on pH alone.']);
  else if(ph>8.4) cards.push(['pH — high/alkaline screening flag','Verify calibration and repeat. Test alkalinity/bicarbonate and relevant chemistry. Acidification may be considered only after water demand is tested and a qualified specialist determines a safe dose. Do not add acid based on pH alone.']);
  else cards.push(['pH — within preliminary screening band','Continue monitoring with a calibrated meter. This is a screening band, not proof the water is suitable for every crop or soil.']);
  if(tds>2000) cards.push(['TDS — high salinity screening flag','Confirm using an EC meter and laboratory analysis, including sodium/SAR and chloride where relevant. Assess verified lower-salinity blending or suitable desalination only after testing. Leaching is not automatically safe; check drainage, soil and crop tolerance first.']);
  else if(tds>450) cards.push(['TDS — elevated screening flag','Confirm with EC and laboratory analysis. Check sodium/SAR, chloride and crop/soil conditions before deciding on blending or treatment. TDS alone cannot identify which salts are present.']);
  else cards.push(['TDS — lower screening flag','Continue monitoring and consider confirming EC. Low TDS alone does not establish overall irrigation suitability.']);
  if(temp>35) cards.push(['Temperature — high-temperature check','Check the probe against a reference thermometer and measure away from direct sunlight. Shade the storage tank where practical and recheck before irrigation. Temperature alone does not measure chemical quality.']);
  else if(temp<10) cards.push(['Temperature — low-temperature check','Verify the sensor and repeat at the same sampling point. If practical, allow stored water to approach local ambient temperature; do not heat irrigation water just to reach a universal target.']);
  else cards.push(['Temperature — no flag from this prototype','Record sampling time and location and continue monitoring. Appropriate temperature depends on local conditions and crop/irrigation context.']);
  out.innerHTML='<div style="display:grid;grid-template-columns:repeat(auto-fit,minmax(230px,1fr));gap:12px">'+cards.map(c=>'<div class="card" style="padding:14px"><h3 style="margin-top:0">'+c[0]+'</h3><p>'+c[1]+'</p></div>').join('')+'</div><div class="warning" style="margin-top:12px"><b>Safety note:</b> These are preliminary screening bands, not universal coconut-irrigation limits. No chemical dose is prescribed. Use laboratory results and qualified agricultural/water advice before treatment or major irrigation decisions.</div>';
}
function clearImprovementAdvice() {
  ['advisor-ph','advisor-tds','advisor-temp'].forEach(id=>document.getElementById(id).value='');
  document.getElementById('advisor-results').innerHTML='';
}
</script>

</body></html>
"""

def get_db_connection():
    """Open SQLite database, creating the parent directory if needed."""
    db_file = Path(DB_PATH)
    db_file.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(db_file), timeout=20)
    conn.execute("PRAGMA journal_mode=WAL")
    return conn


def init_db():
    with get_db_connection() as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS prediction_history (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                run_timestamp TEXT NOT NULL,
                sample_id TEXT,
                ph REAL NOT NULL,
                tds_ppm REAL NOT NULL,
                temperature_c REAL NOT NULL,
                predicted_quality TEXT NOT NULL,
                recommendations TEXT NOT NULL,
                model_probabilities_percent TEXT
            )
        """)
        conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_prediction_history_timestamp
            ON prediction_history(run_timestamp)
        """)


def save_history(results, run_timestamp):
    """Append every predicted sample; never overwrite earlier runs."""
    records = []
    for _, row in results.iterrows():
        records.append((
            run_timestamp,
            str(row.get("Sample_ID", "")),
            float(row["pH"]),
            float(row["TDS_ppm"]),
            float(row["Temperature_C"]),
            str(row["Predicted_Quality"]),
            str(row["Recommendations"]),
            str(row.get("Model_Probabilities_Percent", ""))
        ))
    with get_db_connection() as conn:
        conn.executemany("""
            INSERT INTO prediction_history
            (run_timestamp, sample_id, ph, tds_ppm, temperature_c,
             predicted_quality, recommendations, model_probabilities_percent)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """, records)


def read_history(limit=500):
    with get_db_connection() as conn:
        return conn.execute("""
            SELECT run_timestamp, sample_id, ph, tds_ppm, temperature_c, predicted_quality
            FROM prediction_history
            ORDER BY id DESC LIMIT ?
        """, (limit,)).fetchall()


def history_csv_bytes():
    with get_db_connection() as conn:
        history_df = pd.read_sql_query("""
            SELECT id, run_timestamp AS Timestamp, sample_id AS Sample_ID,
                   ph AS pH, tds_ppm AS TDS_ppm, temperature_c AS Temperature_C,
                   predicted_quality AS Predicted_Quality,
                   recommendations AS Recommendations,
                   model_probabilities_percent AS Model_Probabilities_Percent
            FROM prediction_history ORDER BY id ASC
        """, conn)
    return history_df.to_csv(index=False).encode("utf-8")


def get_recommendations(ph, tds, temp):
    tips = []
    if ph < 6.5:
        tips.append("Low pH flag: verify with a calibrated meter and test alkalinity before choosing treatment.")
    elif ph > 8.4:
        tips.append("High pH flag: test alkalinity and bicarbonate to investigate the cause.")
    else:
        tips.append("pH is within the preliminary 6.5–8.4 screening range; this alone does not establish suitability.")
    if tds > 2000:
        tips.append("High TDS flag: confirm electrical conductivity and assess crop tolerance, soil salinity and drainage.")
    elif tds > 450:
        tips.append("Elevated TDS screening flag: confirm electrical conductivity and assess crop/soil conditions.")
    else:
        tips.append("No high-TDS flag under this prototype's simple screening rule.")
    if temp > 35:
        tips.append("High temperature reading: verify the sensor and compare with site conditions.")
    tips.append("Sodium, SAR, chloride, hardness and bicarbonate require additional tests; they are not inferred from these sensors.")
    return " ".join(tips)

def render_page(**kwargs):
    defaults = {"error": None, "warning": None, "result": False, "history": read_history(), "trend_rows": read_history(limit=100)}
    defaults.update(kwargs)
    return render_template_string(PAGE, **defaults)

@app.route("/", methods=["GET", "POST"])
def index():
    global last_output, last_run
    if request.method == "GET":
        return render_page(history=read_history(), trend_rows=read_history(limit=100))
    if "training" not in request.files or "readings" not in request.files:
        return render_page(error="Please upload both CSV files.")
    try:
        tf, rf = request.files["training"], request.files["readings"]
        if not tf.filename.lower().endswith(".csv") or not rf.filename.lower().endswith(".csv"):
            return render_page(error="Please upload CSV files only.")
        train, readings = pd.read_csv(tf), pd.read_csv(rf)
        required_train = FEATURES + [TARGET]
        missing_train = [c for c in required_train if c not in train.columns]
        missing_readings = [c for c in FEATURES if c not in readings.columns]
        if missing_train:
            return render_page(error=f"Training CSV is missing columns: {missing_train}")
        if missing_readings:
            return render_page(error=f"Readings CSV is missing columns: {missing_readings}")
        for col in FEATURES:
            train[col] = pd.to_numeric(train[col], errors="coerce")
            readings[col] = pd.to_numeric(readings[col], errors="coerce")
        train = train.dropna(subset=required_train).copy()
        readings = readings.dropna(subset=FEATURES).copy()
        if train.empty or readings.empty:
            return render_page(error="One of the files has no valid data rows.")
        train[TARGET] = train[TARGET].astype(str).str.strip()
        train = train[train[TARGET] != ""]
        if train[TARGET].nunique() < 2:
            return render_page(error="Training CSV needs at least two different Quality labels.")
        if len(train) < 4:
            return render_page(error="Please provide at least four valid labelled training rows.")
        warning = None
        if "Data_Source" in train.columns and train["Data_Source"].astype(str).str.contains("ILLUSTRATIVE|SIMULATED|DEMO", case=False).any():
            warning = "This training file contains illustrative, simulated or demo data, not verified laboratory results. The run demonstrates the software only; predictions are not scientifically reliable."
        elif len(train) < 30:
            warning = "Small training dataset. Predictions and any accuracy estimate may be unreliable; collect independent, laboratory-labelled samples."
        X, y = train[FEATURES], train[TARGET]
        accuracy = "Not evaluated (small dataset)"
        counts = y.value_counts()
        if len(train) >= 10 and counts.min() >= 2:
            try:
                Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.25, random_state=42, stratify=y)
                test_model = RandomForestClassifier(n_estimators=200, random_state=42, class_weight="balanced")
                test_model.fit(Xtr, ytr)
                accuracy = f"{accuracy_score(yte, test_model.predict(Xte))*100:.1f}%"
            except ValueError:
                pass
        model = RandomForestClassifier(n_estimators=200, random_state=42, class_weight="balanced")
        model.fit(X, y)
        results = readings.copy()
        results["Predicted_Quality"] = model.predict(readings[FEATURES])
        results["Recommendations"] = results.apply(lambda r: get_recommendations(float(r["pH"]), float(r["TDS_ppm"]), float(r["Temperature_C"])), axis=1)
        # These are transparent screening indicators, not extra sensor measurements or model inputs.
        results["Estimated_EC_uS_cm"] = (results["TDS_ppm"] / 0.64).round(0)
        results["EC_Estimate_Note"] = "Rough estimate only: TDS_ppm / 0.64; confirm with an EC meter"
        results["pH_Screening_Flag"] = results["pH"].apply(lambda v: "Below preliminary band" if v < 6.5 else ("Above preliminary band" if v > 8.4 else "No flag under prototype rule"))
        results["TDS_Screening_Flag"] = results["TDS_ppm"].apply(lambda v: "High screening flag" if v > 2000 else ("Elevated screening flag" if v > 450 else "No high-TDS flag under prototype rule"))
        results["Model_Probabilities_Percent"] = [
            json.dumps({str(label): round(float(prob)*100, 1) for label, prob in zip(model.classes_, probs)})
            for probs in model.predict_proba(readings[FEATURES])
        ]
        last_output = results.to_csv(index=False).encode("utf-8")
        last_run = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        save_history(results, last_run)
        preferred = (["Sample_ID"] if "Sample_ID" in results.columns else []) + FEATURES + ["Estimated_EC_uS_cm", "pH_Screening_Flag", "TDS_Screening_Flag", "Predicted_Quality", "Recommendations"]
        rows = results[preferred].fillna("").astype(str).values.tolist()
        summaries = []
        for col, name, unit, note in [
            ("pH", "Average pH", "", "pH units; check calibration"),
            ("TDS_ppm", "Average TDS", " ppm", "Approximate dissolved-solids reading"),
            ("Temperature_C", "Average temperature", " °C", "Measured sample temperature")]:
            val = float(readings[col].mean())
            summaries.append({"name": name, "value": f"{val:.2f}{unit}", "width": max(3, min(100, val / max(float(readings[col].max()), 1) * 100)), "note": note})
        cat_counts = results["Predicted_Quality"].value_counts()
        categories = [{"name": str(k), "count": int(v), "width": round(int(v)/len(results)*100)} for k, v in cat_counts.items()]
        return render_page(result=True, warning=warning, n_train=len(train), n_readings=len(readings),
                           accuracy=accuracy, headers=preferred, rows=rows, summaries=summaries,
                           categories=categories, run_time=last_run)
    except Exception as exc:
        return render_page(error=f"Could not process files: {exc}")

@app.route("/download")
def download():
    if last_output is None:
        return "Run an analysis first, then download results.", 400
    return send_file(BytesIO(last_output), mimetype="text/csv", as_attachment=True, download_name="aquacrop_results.csv")

@app.route("/history/download")
def download_history():
    data = history_csv_bytes()
    return send_file(
        BytesIO(data), mimetype="text/csv",
        as_attachment=True, download_name="aquacrop_prediction_history.csv"
    )


@app.route("/history/clear", methods=["POST"])
def clear_history():
    # Only remove saved prediction records; retain the database and all other app data.
    with get_db_connection() as conn:
        conn.execute("DELETE FROM prediction_history")
    return redirect(url_for("index", history_cleared=1))


init_db()

if __name__ == "__main__":
    print("Open http://127.0.0.1:5000 in your browser.")
    app.run(debug=False, host="0.0.0.0", port=int(os.environ.get("PORT", "5000")))
