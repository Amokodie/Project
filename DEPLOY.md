# Deploy this app (public link)

I cannot log into your accounts for you. Use **Streamlit Community Cloud** (free) to get a shareable URL.

## Prerequisites

- A **GitHub** account.
- This folder pushed to a **public** GitHub repository (private repos need a paid Streamlit plan).

## Steps

1. **Create a repo** on GitHub (e.g. `cmapss-phm-dashboard`) and **do not** commit `.venv` (it is listed in `.gitignore`).

2. From this project folder, run (replace `YOUR_USER` / `YOUR_REPO`):

   ```powershell
   cd "C:\Users\ROG\Desktop\AI Tutorials\Ass2"
   git init
   git add app.py analysis_pdf.py cmapss_data.py eda_charts.py requirements.txt .gitignore .streamlit assets data DEPLOY.md
   git commit -m "C-MAPSS PHM Streamlit dashboard"
   git branch -M main
   git remote add origin https://github.com/YOUR_USER/YOUR_REPO.git
   git push -u origin main
   ```

3. Open **[Streamlit Community Cloud](https://share.streamlit.io/)** and sign in with GitHub.

4. **New app** → pick your repo → **Main file path:** `app.py` → **Deploy**.

5. After a few minutes you get a URL like `https://YOUR_APP.streamlit.app` to share.

## Data on the cloud

- The hosted machine **does not** have your `E:\` drive. The app reads only what you **commit** under the repo’s **`data/`** folder (same layout as NASA: `train_FD00x.txt`, `test_FD00x.txt`, `RUL_FD00x.txt`).
- The dropdown **only lists datasets that exist** in that folder. If you only committed `test_FD001.txt`, only **FD001** appears—**FD002–FD004** are not magic; you must add those files to `data/` and **push** to GitHub, then redeploy.
- GitHub warns above **100 MB per file**; the NASA text files are usually fine, but check repo size limits for your account.

## Optional: secrets

If you ever need API keys, use **Streamlit Cloud → App settings → Secrets**, not git.
