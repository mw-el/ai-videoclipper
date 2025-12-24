# Claude Integration

## Ziel
Ein voll integriertes Claude‑Terminal direkt in der App, das den Kontext der aktiven Session kennt:
- welches Video gerade bearbeitet wird
- welche SRT-Datei dazu gehört
- Prompts auf Basis dieser Daten ausführen kann (z. B. Scene Detection)
- zusätzlich freie Prompts erlaubt, die der Nutzer direkt im Terminal eingibt

## Umsetzung (aktueller Stand)
- **Eingebettetes Terminal**: Das Claude‑CLI läuft in einem integrierten Terminal‑Panel im linken „Claude“-Bereich.
- **Kontextdateien**: Beim Setzen von Video/SRT wird ein Kontextordner erzeugt:
  - `CLAUDE.md` mit Session‑Infos (Video/SRT)
  - `.claude/settings.local.json` mit erlaubten Pfaden
  - `session.json` mit minimalen Metadaten
- **Arbeitsverzeichnis**: Claude startet im Kontextordner (z. B. `.claude-context`).
- **Prompt‑Button**: Ein Icon‑Button sendet den Scene‑Detection‑Prompt in die laufende Claude‑Session.
  - Der Button ist deaktiviert, bis eine SRT verfügbar ist.
- **Manueller Modus**: Der Nutzer kann im Terminal jederzeit eigene Prompts eingeben.

## Dateien
- `claude_panel.py`: Panel‑UI + integriertes Terminal + Kontextlogik.
- `scene-detection-prompt.txt`: Default‑Prompt für Scene Detection.

## Weiterer Ausbau (geplant)
- Kontextdaten erweitern (z. B. SRT‑Auszüge oder Segment‑Metadaten).
- Stabileres Terminal‑Rendering (vollständige TUI‑Emulation, falls nötig).
- Weitere Prompt‑Buttons (z. B. „Scene Selection“, „Summary“).

## Referenz (Muster‑Installation)
- Implementiert in: `/home/matthias/_AA_TipTapAi`
