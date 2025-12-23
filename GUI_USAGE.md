# AI VideoClipper - New GUI Usage Guide

## Overview

The new GUI provides a powerful clip editing interface with synchronized views:
- **LEFT**: Narrow clip list (200-250px) showing detected clips
- **RIGHT**: Full SRT transcript viewer with interactive editing

## Layout

```
┌────────────────────────────────────────────────────┐
│ [Select File]                    [Status: idle]    │
├──────────────┬─────────────────────────────────────┤
│ CLIPS (Left) │ [📍 Set Start][📍 Set End]          │ ← Toolbar
│              │ [⧉ Dup][➗ Split][💾 Export All]    │
│ Clip 1       │                                     │
│ 00:00-00:42  │ SRT Viewer (Green highlight shows  │
│ Segs 1-5     │ active clip's segment range)       │
│ ────────────┤                                     │
│ Clip 2       │ Full transcript scrolls to show    │
│ 00:42-01:28  │ clip start segment at top          │
│ Segs 6-10    │                                     │
│ ────────────┤                                     │
│ ... (6 clips)│                                     │
│              │                                     │
│ [➕ New]     │                                     │
│ [🗑️ Delete]  │                                     │
│              │                                     │
│ Output: ...  │                                     │
└──────────────┴─────────────────────────────────────┘
```

## Workflow: Basic Usage

### 1. Select a Video
- Click **"Select File"** button
- Choose video from your system
- Transcription starts automatically
- Status shows "Status: transcribing"

### 2. Wait for Transcription
- Watch logs at bottom for `[PROGRESS]` messages
- When complete: status shows "Status: X clips found"
- Clips appear in left panel automatically
- Transcript displays in right SRT viewer

### 3. View Clips
- Clips listed on left with:
  - Clip number (1-6)
  - Start and end times
  - Segment range (e.g., "Segs 1-5")
- Click any clip to select it
- **Green highlight** in SRT viewer shows that clip's segments
- Viewer auto-scrolls to show first segment at top

## Editing Clips

### View Details of a Clip
1. Click clip in left panel
2. Green highlight in right panel shows its transcript segments
3. Each segment is clickable

### Set Start Boundary
1. Click on a **segment** in the transcript (right panel)
2. Click **"📍 Set Start"** button (toolbar, top-right)
3. Clip's start time moves to that segment
4. Green highlight updates immediately
5. Log shows: `[CLIP_EDIT] Set clip start to segment X at Ys`

### Set End Boundary
1. Click on a **segment** in the transcript (right panel)
2. Click **"📍 Set End"** button (toolbar, top-right)
3. Clip's end time moves to that segment
4. Green highlight updates immediately
5. Log shows: `[CLIP_EDIT] Set clip end to segment X at Ys`

### Duplicate a Clip
1. Click clip in left panel to select it
2. Click **"⧉ Duplicate"** button (toolbar)
3. Exact copy created with same boundaries
4. New clip appears at the end of list
5. Log shows: `[CLIP_EDIT] Duplicated clip X`

### Split a Clip
1. Click clip in left panel to select it
2. Click on a **segment** in transcript where you want to split
3. Click **"➗ Split"** button (toolbar)
4. Clip splits into 2 at that segment:
   - First part: original start → split point
   - Second part: split point → original end
5. Both clips appear in list
6. Log shows: `[CLIP_EDIT] Split clip X at segment Y`

### Delete a Clip
1. Click clip in left panel to select it
2. Click **"🗑️ Delete"** button (left panel, bottom)
3. Clip is removed from list
4. Log shows: `[CLIP_EDIT] Deleted clip X`

### Create New Custom Clip
1. Click **"➕ New"** button (left panel, bottom)
2. Dialog opens with spinboxes:
   - **Start Segment**: Pick first segment
   - **End Segment**: Pick last segment
3. Click **"Create"** button
4. New clip created with:
   - Time range from selected segments
   - Text from all segments combined
5. New clip appears at end of list
6. Log shows: `[CLIP_EDIT] Created new clip from segments X-Y`

## Exporting Clips

### Export All Clips
1. Click **"💾 Export All"** button (toolbar, top-right)
2. Status shows "Status: exporting all clips"
3. All clips exported to output folder with:
   - Individual subfolders named after clip text
   - `clip_XX.mp4` video file
   - `clip_XX.srt` subtitle file
4. Log shows progress for each clip
5. Status shows "Status: export complete (/path/to/folder)"

### Output Structure
```
/path/to/your/videos/
└── my_video_clips/           ← Auto-created from video name
    ├── Introduction_to_topic/
    │   ├── clip_01.mp4       ← Video segment
    │   └── clip_01.srt       ← Subtitles (starts at 0:00)
    ├── Main_concepts/
    │   ├── clip_02.mp4
    │   └── clip_02.srt
    └── Conclusion_summary/
        ├── clip_03.mp4
        └── clip_03.srt
```

## Tips & Tricks

### Segment Selection
- **Click once** on a segment in the transcript to select it
- The segment's text will be highlighted
- Status bar doesn't change, but segment is "remembered" for Set Start/End
- You can click different segments without performing an action

### Auto-Scroll
- When you select a clip, the SRT viewer automatically scrolls
- The **first segment** of the clip appears at or near the top
- This helps you quickly locate long clips

### Transcript Navigation
- Scroll manually in the SRT viewer to see other parts
- Clicking segments updates the "remembered" segment (for Set Start/End)
- Green highlight stays on selected clip's range

### Quick Edits
- **Duplicate + Edit**: Duplicate a clip, then Set Start/End to create variations
- **Split + Delete**: Split a clip, then delete the unwanted half
- **Set Start/End + Export**: Adjust boundaries, then export to finalize

### Keyboard Shortcuts
- **Ctrl+O**: Open file (same as "Select File" button)
- **Escape**: Close dialogs

## Logging

All clip operations are logged with `[CLIP_EDIT]` prefix:

```
[CLIP_EDIT] User clicked segment 5
[CLIP_EDIT] Set clip start to segment 5 at 12.5s
[CLIP_EDIT] Set clip end to segment 8 at 18.3s
[CLIP_EDIT] Duplicated clip 1
[CLIP_EDIT] Split clip 2 at segment 10
[CLIP_EDIT] Deleted clip 3
[CLIP_EDIT] Created new clip from segments 1-4
```

View logs with:
```bash
tail -f logs/ai_videoclipper.log | grep "\[CLIP_EDIT\]"
```

## Troubleshooting

### Green highlight not showing
- Make sure you've selected a clip (click in left panel)
- Clip must have valid segment indices
- Try scrolling in the SRT viewer

### Set Start/End button doesn't work
- Did you click a segment in the transcript? (Must click in text area)
- Segment must be within the current clip's range
- Check logs for `[CLIP_EDIT]` messages

### Clip doesn't export with correct boundaries
- Make sure you see the green highlight in SRT viewer
- Verify times in left panel updated after Set Start/End
- Check output folder structure was created correctly

### Dialog closes without creating clip
- Make sure Start Segment ≤ End Segment
- Make sure both values are within valid range (0 to last segment)
- Dialog auto-adjusts spinbox values to keep them ordered

## Next Steps

After creating your perfect clips:
1. Click **"💾 Export All"** to create final video files
2. Check output folder for `.mp4` video files
3. Use `.srt` subtitle files with your video player
4. Import clips into your video editor if needed

---

**Remember**: All clip modifications (Set Start/End, Duplicate, Split) happen **in memory**. Changes are only saved to disk when you click **"💾 Export All"**.

To discard all edits, simply select a new video file.
