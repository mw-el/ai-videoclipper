# AI VideoClipper - Desktop Integration

## Desktop Entry Setup

Die Anwendung wurde als Desktop-Eintrag registriert, damit sie über das Anwendungsmenü gestartet werden kann.

### Dateien

- **Desktop-Datei**: `~/.local/share/applications/ai-videoclipper.desktop`
- **Icon**: `/home/matthias/_AA_AI-VideoClipper/videoclipper.png` (212 KB)
- **Launch Script**: `/home/matthias/_AA_AI-VideoClipper/run.sh`

### So starten Sie die App

#### Methode 1: Anwendungsmenü (grafisch)
1. Öffnen Sie das Anwendungsmenü
2. Suchen Sie nach "AI VideoClipper"
3. Klicken Sie auf das Symbol zum Starten

#### Methode 2: Kommandozeile
```bash
/home/matthias/_AA_AI-VideoClipper/run.sh
```

#### Methode 3: Terminal aus dem Verzeichnis
```bash
cd /home/matthias/_AA_AI-VideoClipper
./run.sh
```

### Desktop Entry Eigenschaften

```
Name:        AI VideoClipper
Comment:     Transcribe videos and create clips with AI
Categories:  Multimedia;Video;AudioVideo;
Icon:        videoclipper.png
Terminal:    No (läuft ohne sichtbares Terminal)
Path:        /home/matthias/_AA_AI-VideoClipper
```

### Falls die App nicht im Menü erscheint

1. Stellen Sie sicher, dass die Datei vorhanden ist:
   ```bash
   ls ~/.local/share/applications/ai-videoclipper.desktop
   ```

2. Manuell die Datenbank aktualisieren:
   ```bash
   update-desktop-database ~/.local/share/applications/
   ```

3. Desktop-Umgebung neu laden:
   - Bei KDE: `kbuildsycoca5`
   - Bei GNOME: `killall -9 nautilus`
   - Bei XFCE: `xfdesktop --reload`

### Icon-Größen

Das Icon wird automatisch vom System in verschiedenen Größen angezeigt:
- Anwendungsmenü: 32x32px
- Desktops-Icons: 48x48px oder 64x64px
- Panel: 24x24px oder größer

Die `videoclipper.png` ist in hoher Qualität vorhanden und wird vom System automatisch skaliert.

### Weitere Optionen

Sie können die Desktop-Datei auch direkt editieren:
```bash
nano ~/.local/share/applications/ai-videoclipper.desktop
```

Nützliche Optionen:
- `Terminal=true`: Zeigt Terminal-Fenster während der Ausführung
- `NoDisplay=true`: Versteckt aus dem Menü
- `OnlyShowIn=KDE;` oder `OnlyShowIn=GNOME;`: Nur in bestimmten Desktops anzeigen

---

**Die App ist jetzt vollständig integriert und kann mit einem Klick gestartet werden!**
