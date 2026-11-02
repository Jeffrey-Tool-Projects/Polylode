# Polylode

[English](../../README.md) · [繁體中文](README.zh-TW.md) · [简体中文](README.zh-CN.md) · [日本語](README.ja.md) · [한국어](README.ko.md) · [Español](README.es.md) · [Français](README.fr.md) · **Deutsch**

Polylode fördert die Charaktermodelle aus deinen installierten Spielen zutage — *Poly*gone aus der *Lode* (der Erzader) — und schreibt sie als
texturierte, geriggte Dateien, die sich überall öffnen lassen: glTF (`.glb`), FBX und Blender (`.blend`). Es ist ein Windows-Desktopprogramm mit
Fenster, einer Vorschau, die sich mit der Maus drehen lässt, und einem Modellbetrachter.

| Spiel | Was es liest | Hinweise |
|---|---|---|
| Arknights: Endfield | Alle Prefabs spielbarer Charaktere | Die Asset-Tabellen werden beim ersten Scan aus deiner Installation gebaut (in geprüften, fortsetzbaren Stapeln) und nach einem Update erneut |
| Girls' Frontline 2: Exilium | Alle Outfits spielbarer Charaktere | Nur Wurzeln mit eigenem geskinnten Mesh und vollständigem menschlichem Skelett (keine Haustiere, Requisiten oder leeren Varianten) |
| Wuthering Waves | Alle Anzeigemodelle spielbarer Charaktere | Braucht den aus `tools/wuwa_cli` gebauten Dateileser und deine AES-Schlüsseldatei |
| Vorhandene Konvertierungen | Fertige Modelle in `*_Converted`-Ordnern | Nach Prüfung des SHA-256 jeder Datei kopiert |

## Funktionen

- **Scannen, ankreuzen, exportieren.** Jedes Spiel hat eine Seite: Installation scannen, suchen, Modelle ankreuzen und in einer Warteschlange
  exportieren. Jeder Scan und jeder Export läuft in einem eigenen Prozess, ein fehlschlagender Konverter reißt das Fenster nie mit.
- **Findet deine Installationen.** Vor einem Scan bietet es die Kopien an, die es in deinen Steam-Bibliotheken findet; der gewählte Ordner kann der des
  Spiels sein, der darüber oder einer in seinen Daten.
- **Drei Formate, zwei ohne Blender.** `.glb` und FBX schreibt Polylode selbst (Skelett, geskinnte Meshes, Shape Keys, eingebettete Texturen,
  Transparenz); `.blend` braucht Blender. Ist Blender gesetzt, wird jede Datei zur Kontrolle darin zurückgelesen.
- **Materialien, wie die Spiele sie zeichnen.** Ausschnitte, Transparenz, Lidschatten-Ebenen und Augenglanz-Ebenen werden nach den Regeln geschrieben,
  die aus den Shadern jedes Spiels gelesen wurden. Wo glTF/FBX eine Ebene nicht mischen können, wird ihre Wirkung in die Textur darunter gebacken;
  die `.blend` behält echte Ebenen.
- **Vorschau vor dem Export.** Zeile wählen und Vorschau drücken (oder doppelklicken): Das Modell wird in einen Cache als `.glb` konvertiert und
  neben der Liste gezeigt — linke Maustaste dreht, rechte verschiebt, das Mausrad zoomt.
- **Modellbetrachter.** Listet alle Modelle im Ausgabeordner und öffnet jede `.glb`, `.gltf`, `.fbx` oder `.obj` (hineingezogen, gewählt oder als
  `Polylode.exe <Datei>`, damit es dein „Öffnen mit“-Programm sein kann): Vorder-, Rück-, Seiten- und Draufsicht, Drahtgitter, heller oder dunkler
  Hintergrund, Lichtstärke und der Inhalt eines glTF. Ein gelistetes Modell lässt sich in den Papierkorb löschen.
- **Fünf Sprachen.** English, 繁體中文, 简体中文, 日本語 und 한국어; es folgt dem System, bis du in den Einstellungen eine wählst, und der Wechsel greift sofort.

## Download

Lade das neueste `Polylode-<Version>-windows-x64.zip` von [Releases](https://github.com/JE-Chen/Polylode/releases), entpacke es irgendwo und starte
`Polylode.exe`. Es bringt alles mit: AnimeStudio (für Endfield), eine .NET-10-Laufzeit, den glTF-Importer und ein Software-OpenGL für Rechner ohne
GPU-Treiber.

Voraussetzungen: Windows 10 oder 11, 64 Bit. Blender 4.x oder 5.x nur für `.blend`.

## Benutzung

1. **Einstellungen** — Ausgabeordner und Exportformate angeben. Installationsordner dürfen leer bleiben: Ein Scan fragt danach (und bietet die aus
   Steam an).
2. **Seite eines Spiels** — *Installation scannen*. Der erste Endfield-Scan baut die Asset-Tabellen (Dutzende Minuten, fortsetzbar); der erste
   GF2-Scan indiziert die Bundles. Spätere Scans verwenden beides weiter, bis das Spiel aktualisiert wird.
3. Modelle ankreuzen, *Angekreuzte Modelle exportieren* und unten Warteschlange und Protokoll verfolgen.
4. **Modellbetrachter** — Exportiertes ansehen, vergleichen, löschen.

Einstellungen und Caches liegen in `%LOCALAPPDATA%\Polylode`. Hast du den Extraktor benutzt, als er Teil von IslandLife war, werden beim ersten
Start seine Einstellungen, die Endfield-Asset-Tabellen, der GF2-Index und die Speicher übernommen, nichts wird neu gebaut.

## Aus den Quellen bauen

```powershell
python -m venv .venv
.venv\Scripts\python -m pip install -r requirements-dev.txt
.venv\Scripts\python -m polylode.extractor          # das Fenster aus den Quellen
.venv\Scripts\python -m pytest tests                # Tests
.venv\Scripts\python build.py --dotnet <dotnet-Ordner>   # dist\Polylode\Polylode.exe, danach sein Selbsttest
```

Der Build kompiliert mit Nuitka (ein C-Compiler ist nötig: der von Visual Studio), legt AnimeStudio, eine .NET-10-Laufzeit (`--dotnet`, Standard
`DOTNET_ROOT`) und das Software-OpenGL neben das Programm und startet `Polylode.exe --selftest`, das jeden Konverter, das QML der Vorschau und eine
echte `.glb` lädt; fehlt etwas, schlägt der Build fehl.

## CI/CD

- **CI** (`.github/workflows/ci.yml`) — bei jedem Push und Pull Request nach `main` und `dev`: Lint, Tests und Selbsttest unter Windows.
- **Release** (`.github/workflows/release.yml`) — bei jedem Push nach `main`: Tests, der Nuitka-Build mit dem Selbsttest des gepackten Programms
  und ein GitHub Release mit dem gezippten Programm.

Entwickelt wird auf `dev`; `main` ist das, was veröffentlicht wird.

## Aufbau

```
polylode/extractor/      Fenster, Scans, Aufträge, Vorschau, Betrachter, Sprachen
polylode/endfield/       Arknights: Endfield — Asset-Tabellen, Konvertierung, FBX-Szene
polylode/gf2/            Girls' Frontline 2 — Bundles, spielbare Liste, Konvertierung, Backen der Augenebenen
polylode/wuthering_waves/  Wuthering Waves — Rollentabellen, Zusammensetzen der Anzeigemodelle
polylode/interchange/    die glTF- und FBX-Schreiber (ohne Blender)
tools/                   Blender-Skripte, das Symbol, die FBX-SDK-Prüfung, die Quellen des WuWa-Lesers, AnimeStudio
tests/                   pytest
build.py, main.py        das gepackte Programm
```

## Lizenz und Inhalte

Der Code von Polylode steht unter der GNU LGPL 2.1 (`LICENSE`); das mitgelieferte AnimeStudio unter MIT (`tools/AnimeStudio/LICENSE`). Polylode
enthält keine Spielinhalte: Es konvertiert Dateien aus den Installationen auf deinem eigenen Rechner, und was es schreibt, bleibt dort. Nutze es nur
mit Spielen, die du besitzt, und halte dich an ihre Bedingungen.
