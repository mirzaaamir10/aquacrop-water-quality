# AquaCrop — Render deployment package

## Deploy to Render

1. Create/sign in to a GitHub account at https://github.com.
2. Create a new repository named `aquacrop-water-quality`.
3. Upload `app.py` and `requirements.txt` from this folder to the repository.
4. Open https://dashboard.render.com and connect your GitHub account.
5. Choose **New + → Web Service** and select `aquacrop-water-quality`.
6. Set:
   - Runtime/Language: Python
   - Build Command: `pip install -r requirements.txt`
   - Start Command: `gunicorn app:app`
   - Instance type: Free (if available in your account/region)
7. Click **Deploy Web Service**.
8. Wait for the deployment to finish, then open the `https://...onrender.com` URL shown by Render.

## Notes

- The site lets visitors upload a labelled training CSV and a new readings CSV.
- Required training columns: `pH`, `TDS_ppm`, `Temperature_C`, `Quality`.
- Required readings columns: `pH`, `TDS_ppm`, `Temperature_C`. `Sample_ID` is optional.
- Free web services may sleep after inactivity and take a little time to wake up.
- Uploaded files are processed in memory; the app does not permanently save uploaded CSVs.
- This is a prototype. Use laboratory-validated training data before relying on predictions. pH, TDS and temperature alone cannot establish all irrigation-water hazards.
