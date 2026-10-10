AquaCrop v3.5 — measurement trends + screening indicators + Water Improvement Advisor + saved history

NEW IN V3.5
- Trend charts for saved pH, TDS and temperature readings (latest 100 saved samples).
- Latest saved sensor values shown above the charts.
- Rough estimated EC in µS/cm using TDS / 0.64, clearly labelled as an estimate requiring EC-meter confirmation.
- Result CSV/table now includes estimated EC plus preliminary pH/TDS screening flags.
- The ML model still uses only pH, TDS_ppm and Temperature_C. Derived indicators are not model inputs or direct measurements.

PRESERVED FROM V3.4
- Dashboard and CSV-based model workflow.
- SQLite prediction history and full-history CSV download.
- Clear prediction history button (confirmation required).
- Water Improvement Advisor and Parameter Improvement Guide.

LOCAL RUN
1. Install Python 3.10 or newer.
2. Run: pip install -r requirements.txt
3. Run: python app.py
4. Open http://127.0.0.1:5000

RENDER DEPLOYMENT
1. Replace app.py and requirements.txt in your GitHub repository with these package files.
2. Build command: pip install -r requirements.txt
3. Start command: gunicorn app:app
4. For persistence, mount a Render disk at /var/data (if supported by your plan) and set AQUACROP_DB_PATH=/var/data/aquacrop_history.db.
5. Redeploy and test analysis, charts, CSV export, history download, and Clear prediction history.

SCIENTIFIC LIMITATIONS
The EC value is only a rough conversion estimate: EC (µS/cm) ≈ TDS (ppm) / 0.64. The correct factor varies by water chemistry and calibration. Confirm with an EC meter. Screening bands are preliminary software flags, not universal coconut or crop irrigation limits. pH, TDS and temperature do not directly measure sodium, SAR, hardness, calcium, magnesium, chloride or bicarbonate; use suitable lab tests/sensors. Never present simulated data as field/laboratory data.
