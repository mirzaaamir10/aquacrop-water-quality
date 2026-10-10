from io import BytesIO
import pandas as pd
from flask import Flask, request, render_template_string, send_file
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

app = Flask(__name__)

FEATURES = ["pH", "TDS_ppm", "Temperature_C"]
TARGET = "Quality"
last_output = None

PAGE = r"""
<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>AquaCrop | Water Quality Dashboard</title>
<style>
:root{font-family:Arial,sans-serif;color:#15323b;background:#f3f8f8}
body{margin:0}.hero{background:#0d5360;color:white;padding:28px 5vw}
.hero h1{margin:0 0 8px}.hero p{margin:0;color:#d6eeee}
main{max-width:1000px;margin:26px auto;padding:0 18px}
.card{background:white;border:1px solid #dce8e8;border-radius:16px;padding:22px;margin-bottom:18px;box-shadow:0 5px 18px #173c440d}
h2{margin-top:0;font-size:20px}.grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(220px,1fr));gap:16px}
label{display:block;font-weight:bold;margin-bottom:8px}input[type=file]{width:100%;padding:10px;border:1px solid #c8d8d8;border-radius:8px;box-sizing:border-box}
button,.download{display:inline-block;background:#087f8c;color:white;padding:12px 18px;border:0;border-radius:9px;font-weight:bold;text-decoration:none;cursor:pointer}
button:hover,.download:hover{background:#06636d}.muted{color:#5b7379;font-size:14px}.warning{background:#fff7df;border:1px solid #ecd69a;padding:13px;border-radius:10px;color:#6b5215}
.error{background:#fff0ef;border:1px solid #e6b7b3;padding:13px;border-radius:10px;color:#8c2820}
.metric{background:#edf7f7;border-radius:12px;padding:15px}.metric small{color:#527078}.metric strong{display:block;font-size:26px;margin-top:6px}
table{width:100%;border-collapse:collapse;font-size:14px}th,td{text-align:left;padding:10px;border-bottom:1px solid #e3eded;vertical-align:top}th{background:#f0f7f7}
.table-wrap{overflow-x:auto}.tag{display:inline-block;background:#e3f5ef;padding:5px 8px;border-radius:7px}
footer{text-align:center;color:#6d8589;padding:25px;font-size:13px}
</style>
</head>
<body>
<header class="hero">
<h1>💧 AquaCrop</h1>
<p>Water Quality Prediction & Agricultural Recommendation Dashboard</p>
</header>
<main>
<div class="card">
<h2>Analyse your water samples</h2>
<p class="muted">Upload a labelled training CSV and a CSV of new sensor readings to analyse water quality for agricultural screening.</p>
<form method="post" enctype="multipart/form-data">
<div class="grid">
<div>
<label for="training">Training CSV</label>
<input id="training" type="file" name="training" accept=".csv" required>
<p class="muted">Must contain pH, TDS_ppm, Temperature_C, Quality.</p>
</div>
<div>
<label for="readings">New ESP32 readings CSV</label>
<input id="readings" type="file" name="readings" accept=".csv" required>
<p class="muted">Must contain pH, TDS_ppm, Temperature_C. Sample_ID is optional.</p>
</div>
</div>
<p><button type="submit">Train model and analyse readings</button></p>
</form>
{% if error %}<div class="error">{{ error }}</div>{% endif %}
{% if warning %}<div class="warning">{{ warning }}</div>{% endif %}
</div>
{% if result %}
<div class="card">
<h2>Analysis summary</h2>
<div class="grid">
<div class="metric"><small>Training samples</small><strong>{{ n_train }}</strong></div>
<div class="metric"><small>New readings analysed</small><strong>{{ n_readings }}</strong></div>
<div class="metric"><small>Hold-out accuracy</small><strong>{{ accuracy }}</strong></div>
</div>
<p class="muted">Accuracy is shown only when the dataset allows a small hold-out test. With tiny or illustrative data, it is not a reliable estimate of real-world performance.</p>
</div>
<div class="card">
<h2>Predictions and recommendations</h2>
<div class="table-wrap"><table>
<thead><tr>{% for h in headers %}<th>{{ h }}</th>{% endfor %}</tr></thead>
<tbody>{% for row in rows %}<tr>{% for cell in row %}<td>{{ cell }}</td>{% endfor %}</tr>{% endfor %}</tbody>
</table></div>
<p><a class="download" href="/download">Download results CSV</a></p>
</div>
{% endif %}
<div class="card">
<h2>How to prepare your files</h2>
<p><b>Training CSV:</b> actual sensor readings matched to lab-tested samples, with a Quality label based on a defined irrigation-water assessment.</p>
<p><b>Readings CSV:</b> new readings from your hardware data collection.</p>
<div class="warning"><b>Important:</b> Example labels are not laboratory results. pH, TDS and temperature alone cannot determine sodium, hardness, chloride or all irrigation hazards. Treat recommendations as preliminary screening, not as a treatment prescription.</div>
</div>
</main>
<footer>AquaCrop • Water Quality Prediction & Agricultural Screening</footer>
</body>
</html>
"""


def get_recommendations(ph, tds, temp):
    tips = []
    if ph < 6.5:
        tips.append("Low pH flag: test acidity and alkalinity before choosing treatment.")
    elif ph > 8.4:
        tips.append("High pH flag: test alkalinity and bicarbonate to identify the cause.")
    else:
        tips.append("pH is within the preliminary 6.5–8.4 screening range.")

    if tds > 2000:
        tips.append("High TDS flag: confirm salinity and EC; assess crop tolerance and soil drainage.")
    elif tds > 450:
        tips.append("Elevated TDS flag: check EC, crop salt tolerance, soil salinity and drainage.")
    else:
        tips.append("No high-TDS warning from this preliminary screening.")

    if temp > 35:
        tips.append("High temperature reading: verify sensor and assess crop-specific conditions.")

    tips.append("This system cannot infer sodium, SAR, chloride, hardness or bicarbonate from these three sensors.")
    return " ".join(tips)


def render_page(**kwargs):
    defaults = {"error": None, "warning": None, "result": False}
    defaults.update(kwargs)
    return render_template_string(PAGE, **defaults)


@app.route("/", methods=["GET", "POST"])
def index():
    global last_output
    if request.method == "GET":
        return render_page()

    if "training" not in request.files or "readings" not in request.files:
        return render_page(error="Please upload both CSV files.")

    try:
        training_file = request.files["training"]
        readings_file = request.files["readings"]
        if not training_file.filename.lower().endswith(".csv") or not readings_file.filename.lower().endswith(".csv"):
            return render_page(error="Please upload CSV files only.")

        train = pd.read_csv(training_file)
        readings = pd.read_csv(readings_file)

        required_train = FEATURES + [TARGET]
        missing_train = [c for c in required_train if c not in train.columns]
        missing_readings = [c for c in FEATURES if c not in readings.columns]
        if missing_train:
            return render_page(error=f"Training CSV is missing columns: {missing_train}")
        if missing_readings:
            return render_page(error=f"Readings CSV is missing columns: {missing_readings}")

        train = train.dropna(subset=required_train).copy()
        readings = readings.dropna(subset=FEATURES).copy()
        if train.empty or readings.empty:
            return render_page(error="One of the files has no valid data rows.")
        if train[TARGET].astype(str).nunique() < 2:
            return render_page(error="Training CSV needs at least two different Quality labels.")

        warning = None
        if "Data_Source" in train.columns and train["Data_Source"].astype(str).str.contains("ILLUSTRATIVE_NOT_LAB", case=False).any():
            warning = "Your training file contains illustrative demo rows, not lab results. This run only demonstrates the website/code; predictions are not scientifically reliable."
        elif len(train) < 30:
            warning = "Small training dataset: predictions and accuracy may be unreliable. Use many independent, laboratory-labelled samples."

        X, y = train[FEATURES], train[TARGET].astype(str)
        accuracy = "Not evaluated (small dataset)"
        counts = y.value_counts()
        if len(train) >= 10 and counts.min() >= 2:
            try:
                Xtr, Xte, ytr, yte = train_test_split(
                    X, y, test_size=0.25, random_state=42, stratify=y
                )
                test_model = RandomForestClassifier(
                    n_estimators=200, random_state=42, class_weight="balanced"
                )
                test_model.fit(Xtr, ytr)
                accuracy = f"{accuracy_score(yte, test_model.predict(Xte))*100:.1f}%"
            except ValueError:
                pass

        model = RandomForestClassifier(
            n_estimators=200, random_state=42, class_weight="balanced"
        )
        model.fit(X, y)
        results = readings.copy()
        results["Predicted_Quality"] = model.predict(readings[FEATURES])
        results["Recommendations"] = results.apply(
            lambda r: get_recommendations(
                float(r["pH"]), float(r["TDS_ppm"]), float(r["Temperature_C"])
            ), axis=1
        )
        results["Model_Probabilities_Percent"] = [
            str({str(label): round(float(prob) * 100, 1)
                 for label, prob in zip(model.classes_, probs)})
            for probs in model.predict_proba(readings[FEATURES])
        ]
        last_output = results.to_csv(index=False).encode("utf-8")

        preferred = []
        if "Sample_ID" in results.columns:
            preferred.append("Sample_ID")
        preferred += FEATURES + ["Predicted_Quality", "Recommendations"]
        shown = results[preferred].fillna("").astype(str)
        rows = shown.values.tolist()

        return render_page(
            result=True, warning=warning, n_train=len(train),
            n_readings=len(readings), accuracy=accuracy,
            headers=preferred, rows=rows
        )
    except Exception as exc:
        return render_page(error=f"Could not process files: {exc}")


@app.route("/download")
def download():
    if last_output is None:
        return "Run an analysis first, then download results.", 400
    return send_file(
        BytesIO(last_output), mimetype="text/csv",
        as_attachment=True, download_name="water_quality_results.csv"
    )


if __name__ == "__main__":
    import os
    port = int(os.environ.get("PORT", "5000"))
    app.run(debug=False, host="0.0.0.0", port=port)
