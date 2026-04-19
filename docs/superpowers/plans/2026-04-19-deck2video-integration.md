---
noteId: "f7eb94f03bef11f1a0d1c39625346603"
tags: []

---

# deck2video Integration Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Produce `output/engine_rul_video.mp4` — a 1920x1080, ~3-minute narrated MP4 that walks through our Engine Failure / RUL work on NASA C-MAPSS FD001 using existing artifacts in `output/` and the `deck2video` tool.

**Architecture:** A self-contained `video/` directory holds (1) a Marp markdown deck with HTML-comment speaker notes, (2) a bash build script that invokes `python -m deck2video`, and (3) a README with Windows install steps. The external `deck2video` repo is cloned alongside the project but gitignored. Nothing in `app.py` or `requirements.txt` changes.

**Tech Stack:** Marp (markdown slides), deck2video (slide-to-video orchestrator), Chatterbox (local TTS, no API keys), ffmpeg (video encoding), Node.js + marp-cli (slide rendering), Python 3.11.

---

## File Structure

| File | Responsibility | Status |
|---|---|---|
| `video/engine_rul_video.md` | Marp deck — 10 slides, embedded `output/*.png`, HTML-comment speaker notes | Create |
| `video/build_video.sh` | Single bash command that runs deck2video with correct args | Create |
| `video/README_VIDEO.md` | Windows install + build instructions for teammates | Create |
| `.gitignore` | Add `deck2video/` and `output/engine_rul_video.mp4` | Modify |
| `output/engine_rul_video.mp4` | Final rendered video | Generated (not committed) |
| `deck2video/` | Cloned upstream repo | External (gitignored) |

Boundaries: `video/*` only reads from `output/*.png` and `output/summary.txt`. It does not import any project Python module. The Streamlit app is untouched.

---

## Task 1: Gitignore the external repo and generated artifact

**Files:**
- Modify: `.gitignore`

- [ ] **Step 1: Read current `.gitignore`**

Run: Read `.gitignore` (project root). Note its existing entries.

- [ ] **Step 2: Append deck2video and the generated mp4**

Append these lines to `.gitignore`:

```
# deck2video — cloned upstream tool, not our source
deck2video/

# Generated narrated video (large binary, rebuildable)
output/engine_rul_video.mp4
```

- [ ] **Step 3: Verify the entries apply**

Run: `git check-ignore -v deck2video/ output/engine_rul_video.mp4 || true`
Expected: Both paths print `.gitignore:<line>:<pattern>` — confirming they are ignored. (The mp4 doesn't exist yet; check-ignore still reports the pattern.)

- [ ] **Step 4: Commit**

```bash
git add .gitignore
git commit -m "Gitignore deck2video clone and generated mp4"
```

---

## Task 2: Create the `video/` directory and scaffold the Marp deck

**Files:**
- Create: `video/engine_rul_video.md`

- [ ] **Step 1: Create the directory**

Run: `mkdir -p video`

- [ ] **Step 2: Write the Marp deck header**

Create `video/engine_rul_video.md` with exactly this content (header + first slide only; remaining slides added in later tasks):

````markdown
---
marp: true
theme: default
size: 16:9
paginate: false
backgroundColor: #0e1117
color: #e8e8e8
style: |
  section {
    font-family: 'Inter', 'Segoe UI', sans-serif;
    padding: 60px 80px;
  }
  h1 { color: #ffffff; font-size: 64px; margin-bottom: 0.2em; }
  h2 { color: #f5a623; font-size: 40px; margin-bottom: 0.4em; }
  .metric { color: #4fc3f7; font-weight: 700; font-size: 72px; }
  .caption { color: #a0a0a0; font-size: 22px; }
  img { border-radius: 8px; }
---

<!-- _class: lead -->

# Engine Failure & Remaining Useful Life Prediction

## NASA C-MAPSS · CNN vs Physics-Informed Neural Network

<br>

**Assignment 2 — RCSSTEAP, Beihang University**

Kodie Amo Kwame (LS2525226)
Sumara Alfred Salifu (LS2525245)
Peta Mimi Precious (LS2525255)

<!-- note: Jet engines fail. When they do, lives and millions of dollars are at stake. This is our approach to predicting engine failure before it happens — physics-informed deep learning on NASA's turbofan dataset. -->
````

- [ ] **Step 3: Verify Marp syntax parses (optional, if marp-cli installed)**

Run (Git Bash): `marp --stdin --version` — just confirms marp is callable. Skip if marp-cli is not yet installed; Task 7 handles install.

- [ ] **Step 4: Commit**

```bash
git add video/engine_rul_video.md
git commit -m "Scaffold Marp deck with title card for RUL video"
```

---

## Task 3: Add slides 2–5 (dataset, RUL concept, architectures, physics loss)

**Files:**
- Modify: `video/engine_rul_video.md`

- [ ] **Step 1: Append slides 2 through 5**

Append this content to `video/engine_rul_video.md`:

````markdown

---

## The Dataset — NASA C-MAPSS

![bg right:55% fit](../output/sensor_trajectories.png)

- 4 fault modes: **FD001 – FD004**
- 21 sensors, 3 operating settings
- Engines run to failure
- **FD001**: 100 train / 100 test engines

<!-- note: NASA's C-MAPSS dataset simulates turbofan engine degradation across four fault modes — F D zero zero one through F D zero zero four. Each engine runs until failure, logging twenty one sensors every cycle. We focus on F D zero zero one — one hundred training engines, one hundred test engines. -->

---

## What Is RUL?

![bg right:55% fit](../output/rul_trajectory_example.png)

**Remaining Useful Life** — cycles left before failure.

- Early life: RUL is effectively **constant**
- Degradation becomes observable only in the **final phase**
- We use a **piecewise-linear cap at 125 cycles**

<!-- note: Remaining Useful Life, or R U L, is the number of cycles an engine has left before failure. Early on, R U L is effectively constant — degradation only becomes observable in the final phase. We cap R U L at one hundred twenty five cycles so the model focuses on the degradation regime. -->

---

## Two Models

![bg right:50% fit](../output/network_architectures.png)

**1D-CNN (baseline)**
Conv1D(F=16, k=5) → GAP → Dense(32) → Dense(1)

**PINN (physics-informed MLP)**
Dense(96) → Dense(48) → Dense(1)
**+ physics loss**

<!-- note: We built two models. A baseline one-D convolutional network that learns purely from data — Conv1D, global average pooling, two dense layers. And a physics-informed neural network — a P I N N — that knows R U L must decrease monotonically by exactly one cycle each step. -->

---

## The Physics Loss

The PINN adds two terms on adjacent-cycle pairs:

**Monotonicity:** `ReLU(RUL(t+1) − RUL(t))²`
> RUL should never increase.

**Drift:** `(RUL(t) − RUL(t+1) − 1)²`
> RUL should drop by exactly one per cycle.

**λ_mono = 0.15 · λ_drift = 0.05**

<!-- note: The P I N N adds two physics terms. A monotonicity penalty that discourages predicting R U L going up — because a degraded engine cannot un-degrade. And a drift penalty that enforces a one-cycle decrease between consecutive windows. Lambda mono zero point one five, lambda drift zero point zero five. -->
````

- [ ] **Step 2: Commit**

```bash
git add video/engine_rul_video.md
git commit -m "Add slides 2-5: dataset, RUL, architectures, physics loss"
```

---

## Task 4: Add slides 6–10 (training, results, conclusion)

**Files:**
- Modify: `video/engine_rul_video.md`

- [ ] **Step 1: Append slides 6 through 10**

Append this content to `video/engine_rul_video.md`:

````markdown

---

## Training

![bg right:55% fit](../output/training_curves.png)

- **14,207** training windows, **3,524** validation
- CNN: 18 epochs · PINN: 22 epochs
- **Total wall time: 20.5 s**
- PINN converges to lower val loss thanks to the physics prior

<!-- note: Both models trained in under twenty one seconds on fourteen thousand two hundred seven windows. The P I N N converges to lower validation loss thanks to the physics regularizer. -->

---

## Predictions — CNN vs PINN

![bg left:50% fit](../output/pred_vs_actual_cnn.png)
![bg right:50% fit](../output/pred_vs_actual_pinn.png)

<br><br><br><br><br>

**Left:** CNN — scattered
**Right:** PINN — hugs the diagonal, especially near end of life

<!-- note: On one hundred unseen test engines, the C N N predictions scatter widely. The P I N N predictions hug the diagonal far more tightly — especially near end of life, where early failure is most costly. -->

---

## Results

![bg right:45% fit](../output/model_comparison.png)

|  | CNN | PINN | Δ |
|---|---:|---:|---:|
| **RMSE** | 18.68 | **14.64** | **−21.6%** |
| **MAE** | 13.69 | **11.14** | **−18.6%** |
| **NASA score** | 1107.5 | **338.6** | **−69.4%** |

<!-- note: The numbers: C N N R M S E eighteen point six eight, P I N N fourteen point six four — a twenty one point six percent improvement. On N A S A's asymmetric score function, which punishes late predictions, we drop from eleven hundred seven to three hundred thirty eight — a sixty nine percent reduction. -->

---

## Error Profile

![bg left:50% fit](../output/error_histogram.png)
![bg right:50% fit](../output/per_engine_rul.png)

<br><br><br><br><br>

**Left:** Error distribution — tighter for PINN
**Right:** Per-engine RUL trajectories track ground truth smoothly

<!-- note: Errors are tighter, and per-engine R U L trajectories track the truth line smoothly. No post-hoc smoothing — this comes from the physics prior alone. -->

---

<!-- _class: lead -->

# Physics-Informed Learning
# = Safer, More Accurate RUL

<br>

**Thank you**

Kodie Amo Kwame · Sumara Alfred Salifu · Peta Mimi Precious
RCSSTEAP, Beihang University

<!-- note: Physics-informed learning turned a small network into a safer, more accurate R U L estimator. Thank you. -->
````

- [ ] **Step 2: Verify total slide count**

Run (Git Bash): `grep -c '^---$' video/engine_rul_video.md`
Expected: `11` (1 front-matter terminator + 10 slide separators = 11 occurrences of `---` alone on a line).

Note: if the count is 11, slide count is correct. If not, a slide separator was mistyped — open the file and fix.

- [ ] **Step 3: Commit**

```bash
git add video/engine_rul_video.md
git commit -m "Add slides 6-10: training, results, error profile, conclusion"
```

---

## Task 5: Write the build script

**Files:**
- Create: `video/build_video.sh`

- [ ] **Step 1: Write the script**

Create `video/build_video.sh`:

```bash
#!/usr/bin/env bash
# Build output/engine_rul_video.mp4 from video/engine_rul_video.md via deck2video.
# Run this from the PROJECT ROOT in Git Bash, after `source deck2video/setup.sh`.

set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"

if [ ! -d "deck2video" ]; then
  echo "ERROR: deck2video/ not found. Clone it first:"
  echo "  git clone https://github.com/pjdoland/deck2video"
  exit 1
fi

if [ -z "${VIRTUAL_ENV:-}" ]; then
  echo "ERROR: no virtualenv active. Run:"
  echo "  source deck2video/setup.sh"
  exit 1
fi

mkdir -p output

python -m deck2video \
  video/engine_rul_video.md \
  --output output/engine_rul_video.mp4

echo
echo "Done. Video written to: output/engine_rul_video.mp4"
```

- [ ] **Step 2: Make it executable (best effort on Windows)**

Run (Git Bash): `chmod +x video/build_video.sh`
Expected: no output. (On native Windows `chmod` is a no-op but harmless; Git Bash respects the exec bit.)

- [ ] **Step 3: Smoke-check the script parses**

Run (Git Bash): `bash -n video/build_video.sh`
Expected: no output (syntactically valid).

- [ ] **Step 4: Commit**

```bash
git add video/build_video.sh
git commit -m "Add build_video.sh to drive deck2video"
```

---

## Task 6: Write the README

**Files:**
- Create: `video/README_VIDEO.md`

- [ ] **Step 1: Write the README**

Create `video/README_VIDEO.md`:

````markdown
# How to build the narrated RUL video (Windows 11)

This directory holds the Marp deck and build script that produces
`output/engine_rul_video.mp4` — a ~3-minute narrated MP4 summarizing our
Engine Failure / RUL work.

## One-time setup

### 1. System dependencies

Open **PowerShell as Administrator** and run:

```powershell
winget install Python.Python.3.11
winget install OpenJS.NodeJS.LTS
winget install Gyan.FFmpeg
winget install Git.Git
```

Close and reopen your terminal so PATH updates take effect.

### 2. Marp CLI (renders slide images)

```powershell
npm install -g @marp-team/marp-cli
```

Verify: `marp --version`

### 3. Clone deck2video

From the **project root** (the folder containing `app.py`):

```bash
git clone https://github.com/pjdoland/deck2video
```

### 4. Install deck2video's Python env (Git Bash)

Open **Git Bash** in the project root and run:

```bash
cd deck2video
source setup.sh
cd ..
```

This creates a Python 3.11 venv inside `deck2video/` and installs
Chatterbox TTS plus deck2video's Python dependencies. **First activation
downloads ~2 GB of TTS model weights — be patient.**

## Building the video

From **Git Bash** in the project root, with the deck2video venv still
active:

```bash
bash video/build_video.sh
```

Output: `output/engine_rul_video.mp4` (~3 min, 1920x1080).

First build: ~5–15 min (TTS is CPU-bound).
Rebuilds: ~3–5 min.

## Rebuilding only changed slides

Edit `video/engine_rul_video.md`, then:

```bash
source deck2video/setup.sh   # if venv is no longer active
python -m deck2video video/engine_rul_video.md \
  --output output/engine_rul_video.mp4 \
  --redo-slides 7,8
```

`--redo-slides` accepts comma-separated slide numbers (1-indexed).

## Troubleshooting

| Symptom | Fix |
|---|---|
| `python: command not found` | Reopen Git Bash; check `which python` points into `deck2video/`. Re-run `source deck2video/setup.sh`. |
| `ffmpeg not found` | `winget install Gyan.FFmpeg`, reopen terminal. |
| `marp: command not found` | `npm install -g @marp-team/marp-cli`. |
| Images don't appear on slides | Confirm paths in `engine_rul_video.md` resolve from `video/` — they use `../output/...`. |
| TTS mispronounces "PINN", "FD001" | Speaker notes already spell these phonetically ("P I N N", "F D zero zero one"). Add more phonetic hints in `<!-- note: ... -->` blocks. |
| Chatterbox download hangs | Kill and retry; weights are cached to `~/.cache/` after first successful download. |

## Files

- `engine_rul_video.md` — Marp deck (10 slides + speaker notes)
- `build_video.sh` — build entrypoint
- `../output/engine_rul_video.mp4` — final video (gitignored)
````

- [ ] **Step 2: Commit**

```bash
git add video/README_VIDEO.md
git commit -m "Add README_VIDEO.md with Windows install + build steps"
```

---

## Task 7: Install system dependencies

**Files:** (none — environment only)

> ⚠️ This task requires admin rights and network access. It will be run by the user, not the implementing agent. The agent should surface the exact commands and wait for user confirmation before proceeding to Task 8.

- [ ] **Step 1: User installs Python 3.11, Node.js, ffmpeg via winget**

Ask the user to run in **PowerShell as Administrator**:

```powershell
winget install Python.Python.3.11
winget install OpenJS.NodeJS.LTS
winget install Gyan.FFmpeg
```

- [ ] **Step 2: User installs Marp CLI**

```powershell
npm install -g @marp-team/marp-cli
```

- [ ] **Step 3: Verify installs**

Run (Git Bash, new session):

```bash
python3.11 --version    # expect: Python 3.11.x
node --version          # expect: v20.x or v22.x
npm --version           # expect: 10.x
ffmpeg -version         # expect: ffmpeg version 7.x
marp --version          # expect: @marp-team/marp-cli vX.Y.Z
```

Expected: all five commands print a version string.

- [ ] **Step 4: No commit** (environment changes only)

---

## Task 8: Clone and install deck2video

**Files:** (none in our repo — affects `deck2video/` only, which is gitignored)

- [ ] **Step 1: Clone the upstream repo**

Run (Git Bash, from project root):

```bash
git clone https://github.com/pjdoland/deck2video
```

Expected: `Cloning into 'deck2video'...` then `done.`

- [ ] **Step 2: Run setup**

Run (Git Bash, from project root):

```bash
cd deck2video
source setup.sh
cd ..
```

Expected: venv created, `pip install` output for requirements, shell prompt now shows `(venv)` prefix.

- [ ] **Step 3: Verify deck2video is importable**

Run (Git Bash, same session):

```bash
python -m deck2video --help
```

Expected: prints usage / argument list without ImportError.

- [ ] **Step 4: No commit** (changes confined to gitignored `deck2video/`)

---

## Task 9: Build the video and verify acceptance criteria

**Files:**
- Generated: `output/engine_rul_video.mp4` (gitignored)

- [ ] **Step 1: Run the build**

Run (Git Bash, with deck2video venv active, from project root):

```bash
bash video/build_video.sh
```

Expected: Marp renders 10 slide PNGs, Chatterbox synthesizes 10 audio clips (first run downloads ~2 GB of weights), ffmpeg stitches to MP4. Final line: `Done. Video written to: output/engine_rul_video.mp4`.

- [ ] **Step 2: Verify the file exists and has content**

Run (Git Bash):

```bash
ls -lh output/engine_rul_video.mp4
```

Expected: file size > 5 MB.

- [ ] **Step 3: Verify resolution is 1920x1080**

Run (Git Bash):

```bash
ffprobe -v error -select_streams v:0 \
  -show_entries stream=width,height \
  -of csv=s=x:p=0 \
  output/engine_rul_video.mp4
```

Expected output: `1920x1080`

- [ ] **Step 4: Verify duration is 170–210 seconds (2:50–3:30)**

Run (Git Bash):

```bash
ffprobe -v error -show_entries format=duration \
  -of default=noprint_wrappers=1:nokey=1 \
  output/engine_rul_video.mp4
```

Expected: a number between `170.0` and `210.0`.

If duration is outside the window, adjust speaker notes in `video/engine_rul_video.md` (shorter/longer text) and rerun `bash video/build_video.sh`.

- [ ] **Step 5: Play the video and sanity-check**

Run (Git Bash or Explorer): `start output/engine_rul_video.mp4`

Verify visually:
- All 10 slides appear in order.
- All PNG figures render without clipping.
- Narration is audible and matches the slide on screen.
- Title card shows the three team names and IDs.
- Closing card shows "Thank you".

- [ ] **Step 6: No commit** (mp4 is gitignored)

---

## Task 10: Document the completion

**Files:**
- Modify: `video/README_VIDEO.md` (optionally add a note that the video was successfully generated)

- [ ] **Step 1: Append a "Last built" note to the README (optional)**

Only do this if the user wants a checked-in note. Otherwise skip. Example:

Append to `video/README_VIDEO.md`:

```markdown

## Build history

- 2026-04-19: Initial video built successfully (duration: Xs, size: Y MB).
```

- [ ] **Step 2: Commit (only if Step 1 was done)**

```bash
git add video/README_VIDEO.md
git commit -m "Note initial successful build in README"
```

---

## Acceptance Criteria (from spec)

Verify at end of Task 9:

1. ✅ `output/engine_rul_video.mp4` exists, 1920x1080, plays in VLC / Windows Media Player.
2. ✅ Duration between 170 s and 210 s.
3. ✅ All 10 slides visible with audible narration.
4. ✅ All figures from `output/` render without clipping.
5. ✅ `video/README_VIDEO.md` is sufficient for a teammate to reproduce the build.

## Rollback

If the project needs to revert: `git revert` the commits from Tasks 1–6, delete the `video/` directory, and delete the `deck2video/` clone. `app.py`, `requirements.txt`, and the Streamlit dashboard are never touched, so there is nothing app-side to undo.
