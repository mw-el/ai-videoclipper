# AI VideoClipper - Implementation Guide

## Current Implementation Status

### ✓ Completed Features

1. **PyQt6 Desktop Application**
   - Full GUI with video preview and transcript display
   - Real-time logging window at bottom of app
   - Clip list with individual export buttons
   - Automatic output folder organization

2. **Faster-Whisper Integration**
   - Transcription via faster-whisper conda environment
   - Subprocess-based execution for non-blocking UI
   - Progress callbacks for real-time status updates
   - GPU acceleration support (CUDA, float32)

3. **Clip Discovery**
   - Local candidate generation from audio VAD + transcript
   - Claude scores candidates (semantic packaging only)
   - Final clips are imported as a JSON config (time-based)

4. **Output Organization**
   - Automatic output folder in source video directory
   - Named `{video_name}_clips`
   - Each clip in its own subfolder
   - Subfolder names derived from first 5-6 words of clip text
   - Both MP4 video and SRT subtitle files per clip

5. **Comprehensive Logging**
   - Multi-level logging (console, file, Qt UI)
   - Tagged log entries for easy debugging
   - File location: `logs/ai_videoclipper.log`

## Directory Structure

```
AI-VideoClipper/
├── clips_editor.py              # Main UI application
├── faster_whisper_transcriber.py # Transcription engine
├── clip_model.py                # Clip finding & export
├── scene_detection_pipeline.py  # Scene detection + candidate ranking
├── srt_viewer.py                # Transcript display widget
├── preview_player.py            # Video preview widget
├── srt_utils.py                 # SRT parsing utilities
├── time_utils.py                # Time formatting utilities
├── logger.py                    # Logging configuration
├── environment.yml              # Conda environment definition
├── requirements.txt             # Python dependencies
├── run.sh                       # Launch script
└── logs/                        # Log directory
    └── ai_videoclipper.log     # Application log file
```

## Output Folder Structure Example

When you select a video like `/home/user/Videos/my_lecture.mp4` and export clips:

```
/home/user/Videos/
├── my_lecture.mp4               # Original video
└── my_lecture_clips/            # Auto-created output folder
    ├── Introduction_key_concepts_here/
    │   ├── clip_01.mp4
    │   └── clip_01.srt
    ├── Advanced_topics_explained/
    │   ├── clip_02.mp4
    │   └── clip_02.srt
    └── Conclusion_summary_points/
        ├── clip_03.mp4
        └── clip_03.srt
```

**Key points:**
- Output folder created in same directory as source video
- Subfolders named from transcript text (first 5-6 words)
- Special characters sanitized to underscores
- Each clip has matching video and SRT files

## Logging Architecture

### Log Sources

1. **Transcriber** (`faster_whisper_transcriber.py`)
   - Subprocess calls and conda activation
   - SRT file creation
   - Segment parsing
   - Tagged as `[PROGRESS]`, `[ERROR]`

2. **Worker Thread** (`clips_editor.py`)
   - Thread lifecycle: START → COMPLETE → signal emission
   - Result type and length information
   - Tagged as `[WORKER]`

3. **Signal Processing** (`clips_editor.py`)
   - Callback reception and unpacking
   - Segment count verification
   - Tagged as `[CALLBACK]`

4. **Display Updates** (`clips_editor.py`, `srt_viewer.py`)
   - SRT viewer method calls
   - Segment count confirmation
   - Tagged as `[DISPLAY]`, `[SRT_VIEWER]`

5. **Export Operations** (`clips_editor.py`)
   - Clip folder creation
   - Video and SRT file writing
   - Tagged as `[CLIPS]`

6. **Worker Connections** (`clips_editor.py`)
   - Thread and signal setup verification
   - Tagged as `[RUN_WORKER]`

### Debugging Transcript Display Issues

Follow this trace in the log file:

```
1. [RUN_WORKER] Starting worker thread for _transcribe_and_find_clips
2. [WORKER] START: _transcribe_and_find_clips at HH:MM:SS
3. [PROGRESS] Loading model...
4. [PROGRESS] Model loaded. Starting transcription...
5. ... (transcription logs from faster-whisper) ...
6. [WORKER] COMPLETE: _transcribe_and_find_clips at HH:MM:SS
7. [WORKER] Result type: <class 'tuple'>
8. [WORKER] Result length: 2
9. [WORKER] Emitting finished signal with result...
10. [WORKER] Finished signal emitted
11. [CALLBACK] on_transcription_ready called with payload type: <class 'tuple'>
12. [CALLBACK] Unpacked payload: result=<class 'faster_whisper_transcriber.TranscriptionResult'>, clips=<class 'list'>
13. [CALLBACK] Result has 65 segments, 6 clips found
14. [CALLBACK] First segment: SrtSegment(index=1, start=0.0, end=2.5, text='...')
15. [DISPLAY] Calling srt_viewer.set_segments()...
16. [SRT_VIEWER] set_segments() called with 65 segments
17. [DISPLAY] ✓ set_segments() completed
18. [SRT_VIEWER] Display updated with 65 segments and 195 display lines
19. [CLIPS] Populating clips list...
20. ✓ Ready to export 6 clips
```

If you don't see step 11 onwards, the callback signal isn't being received. Check for exceptions in steps 1-10.

## Testing Workflow

### 1. Test Transcription
```bash
# Terminal 1: Start the app and watch logs
cd /home/matthias/_AA_AI-VideoClipper
./run.sh

# Terminal 2: Monitor logs in real-time
tail -f logs/ai_videoclipper.log
```

### 2. Select Video
- Click "Select File"
- Choose a video from your system
- Watch "Output:" label update with the output path

### 3. Verify Transcription Starts
- Status should change to "Status: transcribing"
- Watch logs for progress messages
- Look for `[PROGRESS]` tags in the log

### 4. Verify Transcript Displays
- After transcription, look for log entries starting at step 11 above
- Transcript should appear in the "Transcript" panel on the right
- Clip count should update in status bar

### 5. Test Clip Export
- Click "Export All Clips" or individual "Export" buttons
- Navigate to the output folder (shown in "Output:" label)
- Verify folder structure:
  - `{video_name}_clips/` folder exists
  - Subfolders for each clip with descriptive names
  - Each subfolder contains `clip_XX.mp4` and `clip_XX.srt`

## Key Configuration Points

### Transcriber Settings
**File:** `clips_editor.py`, line 84

```python
self.transcriber = FasterWhisperTranscriber(
    progress_callback=self._log_progress
)
```

**Settings in** `faster_whisper_transcriber.py`:
- `conda_env`: "fasterwhisper" (must match your conda environment name)
- `language`: "de" (German; change to "en" for English)
- `device`: "cuda" (GPU acceleration)
- `compute_type`: "float32" (precision level)

### Clip Discovery
**File:** `clips_editor.py`, line 218

```python
clips = self.clip_wrapper.find_clips(result.segments, max_clips=6)
```

Change `max_clips=6` to adjust number of clips found.

### Output Directory
**File:** `clips_editor.py`, `_setup_output_dir()` method

Output location is automatically determined:
```python
self.output_dir = video_dir / f"{video_name}_clips"
```

Where `video_dir` is the directory containing your source video.

## Troubleshooting

### Issue: Transcript not showing after transcription completes

**Check log for:**
1. Does step 11 appear? (`[CALLBACK] on_transcription_ready called...`)
2. Does step 18 appear? (`[SRT_VIEWER] Display updated...`)

**If step 11 missing:**
- Worker might be crashing. Check for `[WORKER] ERROR` entries
- Signal connection might be broken. Check `[RUN_WORKER]` entries

**If step 18 missing:**
- Segments might be empty. Check step 13 for segment count
- SRT viewer might be throwing exception. Check `[DISPLAY] Failed to set segments`

### Issue: Transcription takes very long or hangs

- Check if faster-whisper conda environment has CUDA support
- Monitor GPU usage with `nvidia-smi` in another terminal
- Check system RAM - first run downloads large model files

### Issue: Output folder not created

- Check "Output:" label displays correct path
- Verify you have write permissions to video directory
- Check logs for `[RUN_WORKER] Starting worker thread for _export_all_clips`

### Issue: Clip SRT files have wrong timestamps

- This is expected! Clip SRT files start at 0 seconds
- They're trimmed to match the clip duration
- Original timestamps are in the main transcription file

## Recent Changes (This Session)

1. **Comprehensive Logging Added**
   - Worker thread lifecycle logging
   - Signal connection verification
   - Callback and display logging
   - SRT viewer method tracking

2. **Output Folder Restructuring**
   - Moved from `repo/output/clips` to `{video_dir}/{video_name}_clips`
   - Added intelligent subfolder naming from clip text
   - Automatic SRT generation per clip

3. **Smart Clip Naming**
   - Extracts first 5-6 words from clip text
   - Sanitizes for filesystem compatibility
   - Fallback to clip number if text too short

## Next Steps (Optional Enhancements)

- [ ] Add progress bar for transcription
- [ ] Allow custom clip naming scheme
- [ ] Add batch processing for multiple videos
- [ ] Option to combine all clips into single SRT
- [ ] Keyboard shortcuts for common operations
- [ ] Settings dialog for transcriber parameters
- [ ] Theme selection (dark/light)

## Environment Setup

The app requires two conda environments:

**1. AI VideoClipper Environment** (for the app itself)
```bash
conda create -n ai-videoclipper python=3.12 pyqt6 ffmpeg
pip install -r requirements.txt
```

**2. Faster-Whisper Environment** (for transcription)
```bash
conda create -n fasterwhisper python=3.10
conda activate fasterwhisper
pip install faster-whisper torch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia
```

The app activates the fasterwhisper environment automatically via subprocess.
