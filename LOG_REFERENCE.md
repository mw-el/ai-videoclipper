# AI VideoClipper - Log Reference Guide

## Quick Log Message Reference

Use this guide to understand what each log message means and what to look for when debugging.

## Startup Messages

```
Logging to: /home/matthias/_AA_AI-VideoClipper/logs/ai_videoclipper.log
Setting up faster-whisper transcriber...
✓ Transcriber initialized
```

**What it means:** App is starting, transcriber is ready
**Expected:** Should see all three messages when you launch the app

---

## Video Selection Messages

```
Selected video file: /path/to/video.mp4
Output directory will be: /path/to/video_clips
Loading video for preview...
Starting transcription worker...
[RUN_WORKER] Starting worker thread for _transcribe_and_find_clips
[RUN_WORKER] on_finished=on_transcription_ready
[RUN_WORKER] on_error=on_error
[RUN_WORKER] Worker moved to thread
[RUN_WORKER] Connected thread.started -> worker.run
[RUN_WORKER] Connected worker.finished -> on_transcription_ready
[RUN_WORKER] Connected worker.error -> on_error
[RUN_WORKER] Starting thread...
[RUN_WORKER] Thread started (total threads: 1)
```

**What it means:** Video is selected, output folder determined, worker thread starting
**Expected:** All messages should appear in sequence
**If missing:** Check that video file path is valid and readable

---

## Transcription Worker Messages

```
[WORKER] START: _transcribe_and_find_clips at 19:26:55
Starting transcription of: /path/to/video.mp4
Transcribing with language: de
Task: transcribe
```

**What it means:** Worker thread started, transcription beginning
**Expected:** Should see START time, video path, and settings
**If missing:** Worker thread may have crashed; check for ERROR messages

---

## Faster-Whisper Progress Messages

```
[PROGRESS] Loading model...
Temp SRT will be saved to: /tmp/faster_whisper_tmp/transcription_1766515615.srt
Executing faster-whisper subprocess...
Using environment: fasterwhisper
[PROGRESS] Model loaded. Starting transcription...
Calling faster-whisper process (this will take 30-60 seconds)...
[PROGRESS] Process completed
```

**What it means:** Transcriber is loading model and starting transcription
**Expected:** Messages appear over 30-60 seconds
**If stuck:** Model loading might be slow; wait or check CUDA availability

---

## Transcription Completion Messages

```
[WORKER] COMPLETE: _transcribe_and_find_clips at 19:27:25
[WORKER] Result type: <class 'tuple'>
[WORKER] Result length: 2
[WORKER] Emitting finished signal with result...
[WORKER] Finished signal emitted
Transcription completed in 30.5 seconds (return code: 0)
✓ Transcription complete: 65 segments
Finding clips using ClipsAI...
Found 6 clips
```

**What it means:** Transcription succeeded, segments parsed, clips found
**Expected:** Should see COMPLETE, result info, signal emission, and clip count
**If stuck here:** Signal might not be received by main thread

---

## Callback Reception Messages (CRITICAL!)

```
[CALLBACK] on_transcription_ready called with payload type: <class 'tuple'>
[CALLBACK] Unpacked payload: result=<class 'faster_whisper_transcriber.TranscriptionResult'>, clips=<class 'list'>
[CALLBACK] Result has 65 segments, 6 clips found
[CALLBACK] First segment: SrtSegment(index=1, start=0.0, end=2.5, text='...')
```

**What it means:** Callback received from worker thread, payload unpacked successfully
**Expected:** Should see all four messages if transcription was successful
**⚠️ If missing:** UI won't be updated! Check that signal connections are working. Look back for `[RUN_WORKER]` errors

---

## Display Update Messages (CRITICAL!)

```
[DISPLAY] Calling srt_viewer.set_segments()...
[SRT_VIEWER] set_segments() called with 65 segments
[DISPLAY] ✓ set_segments() completed
[DISPLAY] SRT viewer now has 65 segments
[SRT_VIEWER] Display updated with 65 segments and 195 display lines
```

**What it means:** Transcript is being displayed in the UI
**Expected:** Should see all five messages after callback reception
**⚠️ If stuck at "Calling...":** Exception might be thrown in set_segments(). Look for `[DISPLAY] Failed to set segments` error

---

## Clip Population Messages

```
[CLIPS] Populating clips list...
[CLIPS] ✓ Populated 6 clips in list widget
Setting up viewer with 65 segments and 6 clips
✓ Transcript displayed: 65 segments visible
Status: 6 clips found
✓ Ready to export 6 clips
```

**What it means:** Clip list is being populated in the UI
**Expected:** Should see all messages after display update
**If missing:** Callback processing incomplete

---

## Export Messages

```
Exporting clip 1/6...
[DISPLAY] Calling srt_viewer.set_segments()...
Exporting clip {index + 1} video to: /path/to/video_clips/First_words_here/clip_01.mp4
Saving clip {index + 1} SRT to: /path/to/video_clips/First_words_here/clip_01.srt
✓ Exported clip 1 to: /path/to/video_clips/First_words_here
✓ All 6 clips exported to: /path/to/video_clips
Status: export complete (/path/to/video_clips)
```

**What it means:** Clips are being exported
**Expected:** Multiple clip messages for each clip, then completion message
**If missing:** Export might not have started; check status label

---

## Error Messages

### Common Errors

```
ERROR: Transcription failed with code 1
ERROR: SRT file was not created: /tmp/faster_whisper_tmp/transcription_1766515615.srt
ERROR: File not found: /path/to/nonexistent.mp4
```

**What it means:** Transcription subprocess failed
**To fix:** Check file path, available disk space, CUDA status

```
[CALLBACK] Failed to unpack payload: ...
[DISPLAY] Failed to set segments: ...
```

**What it means:** Payload format unexpected or segment processing error
**To fix:** Check that segments are valid SrtSegment objects

```
[WORKER] ERROR in _transcribe_and_find_clips: ...
```

**What it means:** Exception in worker thread
**To fix:** Check following ERROR line for traceback details

---

## Log Entry Format

All log entries include timestamp and level:

```
HH:MM:SS - logger_name - LEVEL - Message text
```

Example:
```
19:27:25 - ai_videoclipper - INFO - ✓ Transcription complete: 65 segments
19:27:26 - ai_videoclipper - ERROR - ERROR: Failed to set segments: IndexError
```

---

## Finding Specific Issues

### "Transcript not showing"

Search log for:
1. `[CALLBACK] on_transcription_ready called` ← Should exist
2. `[SRT_VIEWER] Display updated` ← Should exist
3. If not present, check for `[WORKER] ERROR` or `[DISPLAY] Failed`

### "No clips found"

Search log for:
1. `✓ Transcription complete: X segments` ← Count should be > 0
2. `Found X clips` ← Should be > 0 (or fallback algorithm used)
3. If segments=0, transcription failed
4. If clips=0, ClipsAI failed and fallback returned nothing

### "Stuck on transcribing"

Search log for:
1. `[PROGRESS] Loading model...` ← Check if this appears
2. `[PROGRESS] Model loaded` ← Check if this appears
3. `[WORKER] COMPLETE` ← Check if transcription finishes
4. If stuck before model loaded, CUDA might be missing
5. If stuck after model loaded, transcription in progress (wait or check NVIDIA)

### "Export not working"

Search log for:
1. `Exporting clip 1/6...` ← Should start export
2. `[DISPLAY] Calling srt_viewer.set_segments()...` ← Should show in export process
3. `✓ Exported clip 1 to:` ← Should confirm each clip
4. Check file system directly for output folder

---

## Monitoring Live

### Watch logs in real-time:
```bash
tail -f logs/ai_videoclipper.log
```

### Watch specific log tag:
```bash
tail -f logs/ai_videoclipper.log | grep "\[CALLBACK\]"
```

### Count occurrences:
```bash
grep -c "\[WORKER\]" logs/ai_videoclipper.log
```

### Find all errors:
```bash
grep "ERROR" logs/ai_videoclipper.log
```

### See last transcription:
```bash
tail -100 logs/ai_videoclipper.log | grep -A 50 "\[WORKER\] START"
```

---

## Log File Location

**Path:** `/home/matthias/_AA_AI-VideoClipper/logs/ai_videoclipper.log`

**Created:** Automatically when app starts
**Size:** Grows with each use (consider rotating old logs)
**Permissions:** Readable by current user

---

## Troubleshooting Checklist

- [ ] Check app started with "✓ Transcriber initialized"
- [ ] Check video selected appears in first log message
- [ ] Check output folder path is shown after selection
- [ ] Check `[WORKER] COMPLETE` appears (transcription finished)
- [ ] Check `[CALLBACK] on_transcription_ready called` appears
- [ ] Check `[SRT_VIEWER] Display updated` appears
- [ ] Check `✓ Ready to export X clips` appears
- [ ] Check export starts with `Exporting clip 1/X...`
- [ ] Check `✓ Exported clip 1 to:` appears for each clip
- [ ] Check output folder exists at shown path
- [ ] Check each clip folder contains `clip_XX.mp4` and `clip_XX.srt`

If any step is missing, search for ERROR messages in that section above.
