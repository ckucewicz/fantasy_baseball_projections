# Fantasy Baseball Projections — Setup Guide

This guide walks you through getting the dashboard live on GitHub Pages
from scratch. It should take about 15–20 minutes.

---

## What you'll end up with

- A live URL like `https://yourusername.github.io/fantasy-baseball-projections`
- Projections auto-updated every morning at 6am Central
- You and your co-owner can both open the URL on any phone or browser
- An accuracy tracker tab that fills in automatically each morning

---

## Step 1 — Create the GitHub repository

1. Go to [github.com](https://github.com) and sign in
2. Click the **+** icon in the top right → **New repository**
3. Set the repository name to exactly: `fantasy-baseball-projections`
4. Set visibility to **Public** (required for free GitHub Pages)
5. Leave everything else as default — don't add a README
6. Click **Create repository**

---

## Step 2 — Set up the folder structure locally

Open your terminal and run these commands one at a time:

```bash
# Create the project folder
mkdir fantasy-baseball-projections
cd fantasy-baseball-projections

# Create the subfolders
mkdir -p backend docs .github/workflows

# Initialize git
git init
git branch -M main
```

---

## Step 3 — Add the files

Copy the files you downloaded from Claude into the right locations:

```
fantasy-baseball-projections/
├── .github/
│   └── workflows/
│       └── update_projections.yml
├── backend/
│   ├── projections.py
│   └── requirements.txt
├── docs/
│   └── index.html
├── roster.json
└── SETUP.md
```

**One edit required before continuing:**

Open `docs/index.html` in any text editor and find these two lines
near the top of the `<script>` block (around line 190):

```js
const GITHUB_OWNER = 'YOUR_GITHUB_USERNAME';
const GITHUB_REPO  = 'fantasy-baseball-projections';
```

Replace `YOUR_GITHUB_USERNAME` with your actual GitHub username.
Save the file.

---

## Step 4 — Create a placeholder data.json

GitHub Pages needs `docs/data.json` to exist before the first
projection run, otherwise the dashboard shows an error on first load.
Create it now:

```bash
echo '{
  "generated_at": null,
  "season": 2025,
  "player_count": 0,
  "projections": []
}' > docs/data.json
```

Also create an empty log file for the accuracy tracker:

```bash
echo '[]' > docs/projections_log.json
```

---

## Step 5 — Create a GitHub Personal Access Token

You need a token to push code from your terminal to GitHub.

1. Go to [github.com/settings/tokens](https://github.com/settings/tokens)
2. Click **Generate new token (classic)**
3. Give it a name like `fantasy-baseball`
4. Set expiration to **No expiration** (or 1 year — your call)
5. Check the **`repo`** scope (full control of repositories)
6. Click **Generate token**
7. **Copy the token immediately** — GitHub only shows it once

**Save it somewhere safe** (password manager, notes app).
You'll use it as your password when pushing from terminal.

---

## Step 6 — Push to GitHub

```bash
# Connect your local repo to GitHub
# Replace YOUR_GITHUB_USERNAME with your actual username
git remote add origin https://github.com/YOUR_GITHUB_USERNAME/fantasy-baseball-projections.git

# Stage all files
git add .

# Commit
git commit -m "initial setup"

# Push
git push -u origin main
```

When prompted for credentials:
- **Username:** your GitHub username
- **Password:** the personal access token you just created (not your GitHub password)

---

## Step 7 — Enable GitHub Pages

1. Go to your repo on GitHub
2. Click **Settings** → **Pages** (in the left sidebar)
3. Under **Source**, select **Deploy from a branch**
4. Under **Branch**, select `main` and folder `/docs`
5. Click **Save**

After about 60 seconds your site will be live at:
`https://YOUR_GITHUB_USERNAME.github.io/fantasy-baseball-projections`

GitHub will show you the exact URL on the Pages settings page.

**Share this URL with your co-owner** — that's all they need.

---

## Step 8 — Run your first projection manually

The scheduled automation starts tomorrow morning at 6am Central.
To get data on the dashboard right now:

1. Go to your repo on GitHub
2. Click the **Actions** tab
3. Click **Update Projections** in the left sidebar
4. Click **Run workflow** → **Run workflow**
5. Wait about 45–60 seconds for it to complete (you'll see a green checkmark)
6. Open your dashboard URL and refresh

You should see all your players ranked with projections.

If the workflow fails, click on the failed run to see the error log.
The most common issues are covered in the Troubleshooting section below.

---

## Step 9 — Test it on your phone

Open the dashboard URL on your phone. You should see:
- All players ranked by projected points
- The "last run" timestamp in the top bar
- Tap any player card to expand the breakdown

Bookmark it or add it to your home screen:
- **iPhone:** tap the Share button → Add to Home Screen
- **Android:** tap the three-dot menu → Add to Home Screen

---

## Day-to-day usage

**Every morning:** open the dashboard — projections are already updated
at 6am Central. The top bar shows the exact time of the last run.

**Accuracy tracker:** tap the "Accuracy tracker" tab to see how
yesterday's projections compared to actual points. This fills in
automatically each morning alongside the new projections.

**Export:** on the accuracy tracker tab, use the time filter buttons
then tap "Export CSV" to download a tidy CSV of the projection log
for analysis in R, Python, or Excel.

**When you add or drop a player:**
1. Open `roster.json` in any text editor
2. Add or remove the player entry (see format below)
3. Save the file
4. In terminal:
   ```bash
   git add roster.json
   git commit -m "roster update: added [player name]"
   git push
   ```
5. The next morning's run picks up the change automatically

---

## Roster.json format

When adding a new player, copy this template:

```json
{
  "name": "Full Name",
  "mlb_id": 123456,
  "team": "NYY",
  "bats": "R",
  "positions": ["OF"],
  "status": "active"
}
```

**Finding the MLB ID:**
- Go to [baseball-reference.com](https://baseball-reference.com)
- Search the player, then click through to their Baseball Savant page
- The number in the URL is their MLB ID
- Or Google: `[player name] baseball savant` and grab the ID from the URL

**`bats` values:** `"R"` (right), `"L"` (left), `"S"` (switch hitter)

**`status` values:**
- `"active"` — playing, gets a full projection
- `"dtd"` — day to day, shows grayed out with no projection
- `"il10"` — 10-day IL
- `"il60"` — 60-day IL

**`team`** — use the standard MLB abbreviation:
ARI, ATL, BAL, BOS, CHC, CWS, CIN, CLE, COL, DET,
HOU, KC, LAA, LAD, MIA, MIL, MIN, NYM, NYY, OAK,
PHI, PIT, SD, SEA, SF, STL, TB, TEX, TOR, WSH

---

## How the automation works

GitHub Actions is a free automation tool built into GitHub. The file
`.github/workflows/update_projections.yml` tells GitHub to:

1. Wake up every morning at 6am Central (11:00 UTC in summer,
   12:00 UTC in winter — both cron entries handle daylight saving)
2. Install Python and the required libraries
3. Run `backend/projections.py`
4. Commit the updated `docs/data.json` and `docs/projections_log.json`
   back to the repo
5. GitHub Pages automatically serves the new files

You can also trigger it manually any time from the Actions tab
(useful after probable pitchers are announced mid-morning).

---

## Troubleshooting

**Workflow fails with "ModuleNotFoundError: pybaseball"**
The pip install step failed. Check the Actions log — if it's a
network timeout just re-run the workflow from the Actions tab.

**Dashboard shows "Could not load projections"**
`docs/data.json` doesn't exist or is empty. Make sure you created
the placeholder in Step 4, and that at least one workflow run has
completed successfully.

**Players showing "no game today"**
The MLB Stats API didn't return a game for that team. This is
correct on off days. Double-check that the `team` field in
`roster.json` matches the exact MLB abbreviation (case-sensitive).

**"last run" timestamp shows old time**
GitHub Actions may not have run yet today, or the workflow failed.
Check the Actions tab in your repo to see the status of the last run.

**Projections look off for a new player**
If the player has very few PA this season, the model leans heavily
on career stats. This is intentional — low sample = low confidence.
The projection improves as the season progresses.

**GitHub Pages shows a 404**
Make sure you selected `/docs` as the folder in Pages settings
(not `/ (root)`). Also confirm `index.html` is directly inside
the `docs/` folder, not a subfolder.

---

## Keeping park factors up to date

The park factor table in `projections.py` is hardcoded with
2022–2024 averages. Each spring, update the values in the
`PARK_FACTORS` dict with the latest numbers from:
[fangraphs.com/guts.aspx?type=pf](https://www.fangraphs.com/guts.aspx?type=pf)

This takes about 10 minutes once a year.

---

## Future improvements (v2 ideas)

- Train a simple ML model on the accuracy log to learn optimal
  factor weights rather than hand-setting them
- Add true individual platoon splits per player from FanGraphs
  rather than using league-average L/R adjustments
- Add a "last 7 days actual vs projected" accuracy summary to
  the console log each morning
- Support multiple teams / co-owners with separate rosters

---

*Built with pybaseball, MLB Stats API, and GitHub Actions.*
*Projection model: PA-weighted career/season blend + BABIP adjustment
+ pitcher difficulty (xFIP, K%, BB%, Hard%, WHIP, Stuff+)
+ park factors + pitch type matchup + recent form (3%).*
