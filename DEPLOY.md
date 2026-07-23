# Getting Transit Hunter online — the beginner walkthrough

Goal: a live web link to your working exoplanet app, hosted free. About 15 minutes.
You'll use **Streamlit Community Cloud**, which runs Streamlit apps straight from
a GitHub repo for free.

---

## Step 0 — Test it on your own computer first (recommended)

Before deploying, make sure it runs locally:

```bash
cd exoplanet-app
pip install -r requirements.txt
streamlit run app.py
```

A browser tab opens. Type **Pi Mensae**, click **Find transits**, and you should
see a light curve and a folded transit. The first run downloads data, so give it
a moment. If it works here, it'll work deployed.

---

## Step 1 — Put the app on GitHub (5 min)

1. On github.com, click **+** → **New repository**.
2. Name it **`transit-hunter`**. Set it **Public**. Click **Create repository**.
3. On the repo page, click **uploading an existing file**.
4. Drag in these four files from the `exoplanet-app` folder:
   - `app.py`
   - `detection.py`
   - `requirements.txt`
   - `README.md`
5. Scroll down and click **Commit changes**.

That's your app, versioned online.

---

## Step 2 — Deploy on Streamlit Community Cloud (5 min)

1. Go to **share.streamlit.io** and click **Sign in** → **Continue with GitHub**
   (use the same account). Approve the access it asks for.
2. Click **Create app** → **Deploy a public app from GitHub**.
3. Fill in:
   - **Repository:** `sreejithn68/transit-hunter`
   - **Branch:** `main`
   - **Main file path:** `app.py`
4. Click **Deploy**. It installs the requirements and boots your app — first build
   takes a few minutes (lightkurve + astropy are chunky). Watch the log; when it's
   done you get a live URL like **`https://transit-hunter.streamlit.app`**.

That URL is your shipped product. It goes on your site and your CV.

---

## Step 3 — Link it from your website

Once you have the live URL, tell me and I'll add a **Projects** section to your
site with Transit Hunter as the first entry — a real screenshot, a one-line pitch,
and a "Launch app" button. That's the moment the site stops being About-only and
starts proving you ship.

---

## If something breaks

- **App is "sleeping"** — free apps sleep after inactivity; clicking the link
  wakes it in ~30 seconds. Normal.
- **A specific star fails** — some targets have no light curve, or messy data.
  Try one of the known-good examples (Pi Mensae, Kepler-10) to confirm the app
  itself is fine.
- **Build fails on a package** — paste the error log to me and I'll fix the
  `requirements.txt`.

---

## The honest status to put on your CV

> *"Transit Hunter — a deployed web app that searches real NASA light curves for
> exoplanet transits using Box Least Squares. Live at [link]. Actively extending
> toward an ML classifier and a live data pipeline."*

True, specific, and it links to something that actually runs. That's the whole goal.
