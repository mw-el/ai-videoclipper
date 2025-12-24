# Claude Integration - Usage Guide

## Headless Mode Scene Detection

The Claude panel now runs scene detection in **headless mode** for better reliability and data extraction.

### How It Works

1. **Button Click** → `send_scene_selection_prompt()` triggered
2. **Headless Execution** → `claude -p "prompt" --allowedTools Read,Grep`
3. **Output Parsing** → Extract JSON from Claude response
4. **Signal Emission** → `scene_data_received.emit(scene_data)`
5. **Parent Handles** → Create clips from scene data

### Signal Connection (clips_editor.py)

```python
# In ClipsEditor.__init__() or _create_claude_panel()
self.claude_panel.scene_data_received.connect(self._handle_scene_data)

def _handle_scene_data(self, scene_data: dict):
    """Handle scene cut points from Claude."""
    import logging
    logger = logging.getLogger("ai_videoclipper")

    cut_points = scene_data.get('cut_points', [])
    logger.info(f"[CLAUDE] Received {len(cut_points)} scene suggestions")

    # Convert cut points to clips
    for i, cut_point in enumerate(cut_points):
        timestamp = cut_point.get('timestamp')  # "HH:MM:SS.mmm"
        confidence = cut_point.get('confidence')  # "high/medium/low"
        reason = cut_point.get('reason')

        # Parse timestamp to seconds
        start_time = self._parse_timestamp(timestamp)

        # Create clip (example: 30 second clips starting at each cut point)
        clip = Clip(
            start_time=start_time,
            end_time=start_time + 30.0,  # Adjust as needed
            text=f"Scene {i+1}: {reason}",
            segment_start_index=0,  # Update with actual segment indices
            segment_end_index=0
        )

        self.clips.append(clip)

    # Refresh clip list display
    if hasattr(self, 'clip_list_widget'):
        self.clip_list_widget.set_clips(self.clips)

    # Show confirmation
    logger.info(f"✓ Created {len(self.clips)} clips from Claude suggestions")

def _parse_timestamp(self, ts_str: str) -> float:
    """Convert HH:MM:SS.mmm to seconds."""
    parts = ts_str.split(':')
    hours = int(parts[0])
    minutes = int(parts[1])
    seconds = float(parts[2])
    return hours * 3600 + minutes * 60 + seconds
```

### Expected JSON Format

Claude should return scene data in this format:

```json
{
  "cut_points": [
    {
      "timestamp": "00:01:23.450",
      "confidence": "high",
      "reason": "Visual scene transition from intro to main content",
      "srt_impact": {
        "start_entry": 5,
        "end_entry": 5
      }
    },
    {
      "timestamp": "00:03:45.200",
      "confidence": "medium",
      "reason": "Dialogue break - natural pause point",
      "srt_impact": {
        "start_entry": 12,
        "end_entry": 15
      }
    }
  ]
}
```

### Display Results

Claude's full response (including reasoning) is displayed in the terminal widget:

```
=== Scene Detection Results ===
I analyzed the SRT file and identified the following scene cut points:

1. **00:01:23.450** (High Confidence)
   - Reason: Visual scene transition from intro to main content
   - SRT Entries: 5

2. **00:03:45.200** (Medium Confidence)
   - Reason: Dialogue break - natural pause point
   - SRT Entries: 12-15

```json
{
  "cut_points": [...]
}
```
=== End Results ===
```

### Error Handling

- **Timeout**: 2 minute timeout for Claude execution
- **JSON Parse Errors**: Logged but non-blocking
- **Missing Files**: Status label shows error message

### Advantages Over Interactive Terminal

✓ **Structured Output**: JSON format easy to parse
✓ **No TTY Issues**: Headless mode doesn't need raw mode
✓ **Reliable**: subprocess.run() with timeout
✓ **Clean Integration**: Signal-based architecture
✓ **User Feedback**: Full response shown in terminal widget
