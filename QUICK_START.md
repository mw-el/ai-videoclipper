# AI VideoClipper - Quick Start Guide

## Launch the App

```bash
cd /home/matthias/_AA_AI-VideoClipper
./run.sh
```

The app will start in a few seconds. You'll see:
- PyQt6 window with video preview on the right
- Transcript panel below the preview
- Clips list on the left
- Log window at the bottom
- Output folder info label

## Basic Workflow

### 1️⃣ Select a Video (30 seconds)

1. Click **"Select File"** button
2. Choose a video from your filesystem
3. Watch the **"Output:"** label update with the output folder path
4. Status shows "Status: transcribing"

### 2️⃣ Wait for Transcription (30-60 seconds)

- The transcription subprocess starts automatically
- Watch the **log window** at bottom for progress
- Look for `[PROGRESS]` messages
- App remains responsive during transcription

### 3️⃣ View Results (instant)

When transcription completes, you'll see:
- ✓ Transcript appears in the "Transcript" panel
- ✓ Clip count updates in status bar (e.g., "Status: 6 clips found")
- ✓ Clip list populates on the left

### 4️⃣ Export Clips (2-5 minutes)

**Option A: Export All**
- Click **"Export All Clips"** button
- All clips export to individual folders

**Option B: Export Individual**
- Click **"Export"** button next to any clip in the list
- Single clip exports to its folder

### 5️⃣ Access Your Clips

Navigate to the output folder (shown in "Output:" label):

```
/path/to/your/videos/video_name_clips/
├── First_few_words_from_transcript/
│   ├── clip_01.mp4
│   └── clip_01.srt
├── Another_key_phrase/
│   ├── clip_02.mp4
│   └── clip_02.srt
└── ... (more clips)
```

Each clip folder contains:
- **clip_XX.mp4** - The video segment
- **clip_XX.srt** - Subtitle file with timing

---

## Keyboard Shortcuts

| Shortcut | Action |
|----------|--------|
| `Ctrl+O` | Open file (same as "Select File") |
| `Escape` | Close any dialogs |

## Troubleshooting Quick Fixes

### Issue: "No transcript showing"

**Solution:**
1. Check the log window for errors
2. Look for `[CALLBACK]` messages
3. If not present, transcription failed - check error messages
4. Try a different video file

### Issue: "Export button doesn't work"

**Solution:**
1. Make sure transcription completed (check status bar)
2. Make sure clips were found (count > 0)
3. Make sure output folder path is valid
4. Check you have write permission to that directory

### Issue: "App is slow/freezing"

**Solution:**
1. This is normal during transcription (30-60 seconds)
2. App should remain responsive (can click around)
3. If completely frozen, check CUDA availability
4. Can cancel and try different video

### Issue: "Output folder not created"

**Solution:**
1. Check the "Output:" label shows a path
2. Verify you have write permissions to that directory
3. Try exporting to Desktop or home directory
4. Check system disk space

---

## Example Sessions

### Example 1: 3-Minute Video (German)

```
1. Click "Select File"
2. Choose: ~/Videos/my_lecture.mp4
3. Status: "Status: transcribing"
4. Wait ~40 seconds
5. See: "Status: 6 clips found"
6. Transcript appears with 65 segments
7. Click "Export All Clips"
8. Clips created in: ~/Videos/my_lecture_clips/
9. Find 6 folders with descriptive names
```

**Time:** ~2 minutes total

### Example 2: Export Single Clip

```
1. After transcription, see clip list on left
2. See 6 clips with timestamps
3. Click "Export" button next to clip #2
4. Status: "Status: export complete"
5. Clip created at: ~/Videos/my_lecture_clips/Clip_name/clip_02.mp4
```

**Time:** ~1-2 minutes (depending on clip length)

---

## Important Notes

### About Output Folders

- ✓ Created in **same directory as source video** (not in the app directory)
- ✓ Named `{video_name}_clips` (e.g., `lecture_clips`)
- ✓ One folder per source video
- ✓ Exporting again overwrites previous clips

### About Transcripts

- ✓ Transcription language is German (configurable in code)
- ✓ Full transcript shown in "Transcript" panel during preview
- ✓ Individual SRT files saved per clip (with adjusted timestamps)
- ✓ Timestamps in clip SRT start at 0:00:00

### About Clips

- ✓ Up to 6 clips found per video (configurable)
- ✓ Folder names derived from first 5-6 words of clip text
- ✓ Clips are ranked by relevance (best first)
- ✓ Can preview clip by clicking in transcript

---

## Log File

**Location:** `logs/ai_videoclipper.log`

**Check logs when:**
- Transcript not showing
- Export didn't work
- App crashed
- Something seems slow

**Quick check:**
```bash
# Watch logs in real-time
tail -f logs/ai_videoclipper.log

# Check for errors
grep "ERROR" logs/ai_videoclipper.log

# See last transcription
tail -50 logs/ai_videoclipper.log
```

---

## Settings (In Code)

To change behavior, edit **clips_editor.py**:

### Language (line 213)
```python
result = self.transcriber.transcribe(str(self.video_path))
# Add language parameter:
result = self.transcriber.transcribe(str(self.video_path), language="en")
# Options: "de" (German), "en" (English), "fr" (French), "es" (Spanish), etc.
```

### Max Clips (line 218)
```python
clips = self.clip_wrapper.find_clips(result.segments, max_clips=6)
# Change 6 to any number you want, e.g.:
clips = self.clip_wrapper.find_clips(result.segments, max_clips=10)
```

---

## Common Questions

**Q: Why does transcription take 30-60 seconds?**
A: The first run downloads the transcription model (~2 GB). Subsequent runs use the cached model.

**Q: Can I transcribe English videos?**
A: Yes! Edit line 213 in clips_editor.py and change `language="en"`.

**Q: Where do the clips go?**
A: In a folder called `{video_name}_clips` in the same directory as your video file.

**Q: Can I customize the clip folder names?**
A: Currently they're auto-generated from transcript text. Edit `_get_clip_folder_name()` method to customize.

**Q: What if a clip has no text?**
A: Falls back to `clip_01`, `clip_02`, etc.

**Q: Can I export clips to a different folder?**
A: Not via UI (auto-determined). Edit `_setup_output_dir()` method to change folder location.

**Q: Do I need an internet connection?**
A: Only for the first transcription (to download model). Subsequent transcriptions work offline.

**Q: Can I process multiple videos at once?**
A: Not yet (one at a time). Could be added as future feature.

---

## Getting Help

1. **Check the log file** (`logs/ai_videoclipper.log`)
   - Contains all detailed information
   - Search for `ERROR` for problems

2. **Read the implementation guide** (`IMPLEMENTATION_GUIDE.md`)
   - Complete technical documentation
   - Troubleshooting section with detailed fixes

3. **Check the log reference** (`LOG_REFERENCE.md`)
   - What each log message means
   - How to find specific issues
   - Commands for monitoring logs

4. **Try a test video**
   - Short 1-2 minute video
   - Clear audio (no music)
   - See if it works with simpler input

---

## Quick Checklist Before Starting

- [ ] App launches without errors
- [ ] Video file is readable
- [ ] Transcription environment (fasterwhisper) is installed
- [ ] You have write permissions to video directory
- [ ] System has 4+ GB free RAM
- [ ] (Optional) GPU available and CUDA installed

---

**Ready to go!** 🎬 Select a video and click Start.
