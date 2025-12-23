# AI VideoClipper - Clips Configuration Format

## Overview

The clips configuration system supports two modes of operation:

1. **Auto-define mode**: Automatically detect and create clips using ClipsAI
2. **Manual selection mode**: Manually specify clip boundaries by time or segment numbers

Both modes are configured via JSON files.

---

## File Format: `clips.json`

### Mode 1: Auto-Define Clips (Default)

When you want the app to automatically find clips after transcription:

```json
{
  "mode": "auto",
  "max_clips": 6,
  "description": "Use ClipsAI to automatically detect interesting clips"
}
```

**Parameters:**
- `mode` (string, required): Set to `"auto"`
- `max_clips` (integer, optional): Maximum number of clips to detect. Default: 6
- `description` (string, optional): Human-readable description

---

### Mode 2: Manual Selection by Time

Manually specify clip boundaries using absolute timestamps (in seconds):

```json
{
  "mode": "manual",
  "selection_type": "time",
  "clips": [
    {
      "name": "Kitchen lighting discussion",
      "start_time": 23.199,
      "end_time": 205.12
    },
    {
      "name": "Camera positioning tips",
      "start_time": 412.5,
      "end_time": 658.2
    },
    {
      "name": "Final setup review",
      "start_time": 720.0,
      "end_time": 900.5
    }
  ]
}
```

**Parameters:**
- `mode` (string, required): Set to `"manual"`
- `selection_type` (string, required): Set to `"time"`
- `clips` (array, required): List of clip definitions
  - `name` (string, optional): Descriptive name for the clip
  - `start_time` (number, required): Start time in seconds (float)
  - `end_time` (number, required): End time in seconds (float)

---

### Mode 3: Manual Selection by Segment Numbers

Manually specify clip boundaries using segment indices (1-indexed):

```json
{
  "mode": "manual",
  "selection_type": "segments",
  "clips": [
    {
      "name": "Kitchen lighting discussion",
      "start_segment": 8,
      "end_segment": 36
    },
    {
      "name": "Camera positioning tips",
      "start_segment": 37,
      "end_segment": 60
    },
    {
      "name": "Final setup review",
      "start_segment": 61,
      "end_segment": 65
    }
  ]
}
```

**Parameters:**
- `mode` (string, required): Set to `"manual"`
- `selection_type` (string, required): Set to `"segments"`
- `clips` (array, required): List of clip definitions
  - `name` (string, optional): Descriptive name for the clip
  - `start_segment` (integer, required): Starting segment number (1-indexed)
  - `end_segment` (integer, required): Ending segment number (1-indexed, inclusive)

---

## Usage Workflow

### Step 1: Select Video File
Click "Select File" to choose your video. The app will transcribe it and show segments.

### Step 2: Choose Configuration Method

**Option A: Use Auto-Define (Simplest)**
1. Let ClipsAI automatically find clips
2. Modify them manually as needed
3. Export clips

**Option B: Load Manual Configuration**
1. Create a `clips.json` file with your manual selections
2. Click "Load Clips Config"
3. Select your `clips.json` file
4. The app loads your defined clips
5. Modify them if needed
6. Export clips

### Step 3: Modify (Optional)
- Use "Set Start" and "Set End" buttons to adjust clip boundaries
- Use "Duplicate", "Split", or "Delete" to manage clips
- Use "New" to create new clips from segment ranges

### Step 4: Export
Click "Export All" to export the final clips

---

## Examples

### Example 1: Auto-Define with 8 Clips Maximum

File: `clips.json`
```json
{
  "mode": "auto",
  "max_clips": 8
}
```

### Example 2: Manual Selection by Time

File: `interview_clips.json`
```json
{
  "mode": "manual",
  "selection_type": "time",
  "clips": [
    {
      "name": "Introduction",
      "start_time": 0.0,
      "end_time": 45.5
    },
    {
      "name": "Main topic discussion",
      "start_time": 50.2,
      "end_time": 320.8
    },
    {
      "name": "Q&A section",
      "start_time": 325.0,
      "end_time": 480.5
    },
    {
      "name": "Conclusion",
      "start_time": 485.0,
      "end_time": 520.0
    }
  ]
}
```

### Example 3: Manual Selection by Segments

File: `podcast_clips.json`
```json
{
  "mode": "manual",
  "selection_type": "segments",
  "clips": [
    {
      "name": "Episode intro",
      "start_segment": 1,
      "end_segment": 5
    },
    {
      "name": "Guest introduction",
      "start_segment": 6,
      "end_segment": 12
    },
    {
      "name": "Main interview",
      "start_segment": 13,
      "end_segment": 45
    },
    {
      "name": "Listener questions",
      "start_segment": 46,
      "end_segment": 58
    },
    {
      "name": "Outro",
      "start_segment": 59,
      "end_segment": 65
    }
  ]
}
```

---

## Converting Between Formats

### From Time to Segments
You can see the time range for each segment in the SRT Viewer:
- Segment 8: 23.199s - 32.9s
- Segment 36: 205.12s - 214.8s

Then map: `start_segment: 8, end_segment: 36` instead of `start_time: 23.199, end_time: 205.12`

### From Segments to Time
Look at the first and last segments in your range:
- Segments 8-36 → start_time: 23.199s, end_time: 205.12s

---

## Validation Rules

- `start_time` must be less than `end_time`
- `start_segment` must be less than or equal to `end_segment`
- Segment numbers are 1-indexed (first segment is 1, not 0)
- Time values are in seconds (float)
- `mode` must be either `"auto"` or `"manual"`
- `selection_type` must be `"time"` or `"segments"` (required for manual mode)

---

## File Location

Save your configuration file in any location. When using "Load Clips Config", you can browse to select it.

Suggested locations:
- Next to your video file
- In a dedicated `configs/` folder
- On your desktop

---

## Tips

1. **For interviews**: Use segments - easier to count visually in the UI
2. **For precise editing**: Use time - exact to the millisecond
3. **For batch processing**: Create template JSON files for different video types
4. **Testing**: Start with auto-define, then export/save as JSON to see the format
