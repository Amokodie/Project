---
noteId: "f22ef6f03bf011f1a0d1c39625346603"
tags: []

---

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
| Images don't appear on slides | (1) Confirm paths in `engine_rul_video.md` resolve from `video/` — they use `../output/...`. (2) Marp blocks local file:// by default; `deck2video/deck2video/marp_renderer.py` must pass `--allow-local-files` to the marp invocation. If you re-cloned deck2video, re-apply this patch. |
| `FileNotFoundError: [WinError 2]` on marp | Windows subprocess can't resolve extensionless `marp`. In `deck2video/deck2video/marp_renderer.py`, use `shutil.which("marp.cmd") or shutil.which("marp")` instead of just `"marp"`. |
| TTS mispronounces "PINN", "FD001" | Speaker notes already spell these phonetically ("P I N N", "F D zero zero one"). Add more phonetic hints in `<!-- note: ... -->` blocks. |
| Chatterbox download hangs | Kill and retry; weights are cached to `~/.cache/` after first successful download. |

## Files

- `engine_rul_video.md` — Marp deck (10 slides + speaker notes)
- `build_video.sh` — build entrypoint
- `../output/engine_rul_video.mp4` — final video (gitignored)
