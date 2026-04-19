---
noteId: "93c9bb503bef11f1a0d1c39625346603"
tags: []

---

# deck2video Integration — 3-Minute RUL Narrated Video

Date: 2026-04-19
Owner: Kodie Amo Kwame, Sumara Alfred Salifu, Peta Mimi Precious
Context: Assignment 2, RCSSTEAP, Beihang University

## Goal

Produce a ~3-minute narrated MP4 video (1920x1080) that explains our Engine Failure / Remaining Useful Life (RUL) work on the NASA C-MAPSS FD001 dataset, using the existing trained results in `output/`. Use the `deck2video` tool from https://github.com/pjdoland/deck2video.

## Non-Goals

- No changes to `app.py` or the Streamlit dashboard.
- No retraining of models. Reuse the artifacts already in `output/`.
- No cloud TTS, no API keys. Default local Chatterbox voice only.
- No interactive video-generation feature in the app.

## Source Content

Reuse existing PNG figures from `output/`:

- `sensor_trajectories.png`
- `rul_trajectory_example.png`
- `network_architectures.png`
- `training_curves.png`
- `pred_vs_actual_cnn.png`, `pred_vs_actual_pinn.png`
- `model_comparison.png`
- `error_histogram.png`, `per_engine_rul.png`

Reuse numeric results from `output/summary.txt` / `output/metrics.csv`:

- CNN: RMSE 18.679, MAE 13.689, NASA score 1107.5
- PINN: RMSE 14.644, MAE 11.137, NASA score 338.6
- Relative: RMSE -21.6%, MAE -18.6%, score -69.4%

## Content Arc — 10 Slides, ~3 min 22 s

| # | Visual | Narration (speaker note, abbreviated) | Duration |
|---|---|---|---|
| 1 | Title card (typographic) | Hook: engines fail, lives & money at stake, our approach. | 12 s |
| 2 | `sensor_trajectories.png` | C-MAPSS dataset, 4 fault modes, 21 sensors, run-to-failure. | 22 s |
| 3 | `rul_trajectory_example.png` | Define RUL. Early flat phase. Degradation only observable near end. | 22 s |
| 4 | `network_architectures.png` | Two models: 1D-CNN baseline vs PINN. | 26 s |
| 5 | Text slide — physics loss | Monotonicity penalty + one-cycle-drift penalty. lam=0.15, 0.05. | 22 s |
| 6 | `training_curves.png` | Fast training (<21 s). PINN lower val loss from physics prior. | 18 s |
| 7 | `pred_vs_actual_cnn.png` + `pred_vs_actual_pinn.png` | PINN hugs diagonal, especially near end of life. | 24 s |
| 8 | `model_comparison.png` + big metric numbers | RMSE -21.6%, NASA score -69.4%. | 26 s |
| 9 | `error_histogram.png` + `per_engine_rul.png` | Tighter errors, smoother per-engine trajectories. | 22 s |
| 10 | Closing card (team names + IDs) | "Physics-informed learning — safer, more accurate. Thank you." | 8 s |

Title card text:

```
Engine Failure & Remaining Useful Life Prediction
NASA C-MAPSS · CNN vs Physics-Informed Neural Network

Assignment 2 — RCSSTEAP, Beihang University
Kodie Amo Kwame (LS2525226)
Sumara Alfred Salifu (LS2525245)
Peta Mimi Precious (LS2525255)
```

## Output Artifacts

```
video/
  ├─ engine_rul_video.md       # Marp markdown + HTML-comment speaker notes
  ├─ build_video.sh            # one-liner: python -m deck2video ...
  └─ README_VIDEO.md           # Windows install + build instructions
output/
  └─ engine_rul_video.mp4      # final 1920x1080 narrated MP4 (~3 min)
deck2video/                    # cloned upstream repo (gitignored)
```

`deck2video/` is added to `.gitignore` (it is a cloned external repo, not our source).

## Architecture / Data Flow

```
output/*.png  ──┐
summary.txt    ─┼──▶  engine_rul_video.md (Marp)
branding text  ─┘            │
                             ▼
                    marp-cli (renders slide images)
                             │
                             ▼
              deck2video  ──▶ Chatterbox TTS per slide note
                             │
                             ▼
                    ffmpeg stitch ──▶ output/engine_rul_video.mp4
```

## Dependencies (Windows 11)

System (install once):

- Python 3.11 — `winget install Python.Python.3.11`
- Node.js LTS — `winget install OpenJS.NodeJS.LTS`
- ffmpeg — `winget install Gyan.FFmpeg`
- Marp CLI — `npm i -g @marp-team/marp-cli`
- Git Bash (already present with Git for Windows) — required because `setup.sh` is a bash script.

Python-side (installed by `deck2video/setup.sh` inside its own venv):

- Chatterbox TTS (~2 GB model on first run)
- Other deck2video Python deps per its `requirements.txt`

## Build Steps

1. `git clone https://github.com/pjdoland/deck2video` into project root.
2. In Git Bash: `cd deck2video && source setup.sh` — creates venv, installs deps.
3. In Git Bash: `bash video/build_video.sh` — runs `python -m deck2video video/engine_rul_video.md --output output/engine_rul_video.mp4`.
4. First run downloads Chatterbox (~2 GB). Subsequent rebuilds ~3–5 min on CPU.

## Isolation & Boundaries

- `video/` is self-contained. Only inputs are PNGs in `output/` and the branding text. No Python imports from our app.
- `deck2video/` is treated as a black-box external tool. We do not modify it.
- The Streamlit app is untouched. Dashboard behavior, deploy, and requirements.txt are unaffected.

## Risks / Mitigations

| Risk | Mitigation |
|---|---|
| Chatterbox 2 GB download slow/fails | Document the wait; model is cached after first run |
| Windows + `setup.sh` friction | Require Git Bash; document exact commands |
| Narration length drifts from 3 min | Speaker notes are word-counted (~450 words); `--redo-slides` flag re-renders only changed slides |
| PNG aspect ratios don't fit 16:9 | Marp `![bg fit]` directive; acceptable letterboxing |
| TTS mispronounces "PINN", "FD001", "RMSE" | Use speaker-note phonetic hints ("P I N N", "F D oh oh one") if needed |

## Acceptance Criteria

1. `output/engine_rul_video.mp4` exists, is 1920x1080, and plays in VLC / Windows Media Player.
2. Duration between 2 min 50 s and 3 min 30 s.
3. All 10 slides visible, each with audible narration matching the script.
4. All figures from `output/` render without clipping.
5. `video/README_VIDEO.md` lets a teammate reproduce the build from a clean Windows machine in <60 min (excluding Chatterbox download).

## Out of Scope (Possible Follow-ups)

- Subtitles / SRT export.
- Uploading to YouTube/Bilibili.
- Multilingual narration (Mandarin version for RCSSTEAP context).
- A Streamlit "Generate Video" button.
