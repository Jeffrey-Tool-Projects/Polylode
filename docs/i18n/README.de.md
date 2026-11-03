# Polylode

[English](../../README.md) · [繁體中文](README.zh-TW.md) · [简体中文](README.zh-CN.md) · [日本語](README.ja.md) · [한국어](README.ko.md) · [Español](README.es.md) · [Français](README.fr.md) · **Deutsch**

**Schritt-für-Schritt-Anleitung (Englisch):** [https://je-chen.github.io/Polylode/](https://je-chen.github.io/Polylode/)

Polylode fördert die Charaktermodelle aus deinen installierten Spielen zutage — *Poly*gone aus der *Lode* (der Erzader) — und schreibt sie als
texturierte, geriggte Dateien, die sich überall öffnen lassen: glTF (`.glb`), FBX und Blender (`.blend`). Es ist ein Windows-Desktopprogramm mit
Fenster, einer Vorschau, die sich mit der Maus drehen lässt, und einem Modellbetrachter.

| Spiel | Was es liest | Hinweise |
|---|---|---|
| Arknights: Endfield | Alle Prefabs spielbarer Charaktere | Die Asset-Tabellen werden beim ersten Scan aus deiner Installation gebaut (in geprüften, fortsetzbaren Stapeln) und nach einem Update erneut |
| Girls' Frontline 2: Exilium | Alle Outfits spielbarer Charaktere | Nur Wurzeln mit eigenem geskinnten Mesh und vollständigem menschlichem Skelett (keine Haustiere, Requisiten oder leeren Varianten). Ein Outfit mit austauschbaren Teilen wird so gezeigt, wie das Spiel es standardmäßig anzieht; die anderen Versionen und niedrigeren Detailstufen sind versteckt in FBX und .blend (die .glb enthält nur das Gezeigte: Blender und die meisten Betrachter ignorieren glTFs Versteckt-Markierung) |
| Wuthering Waves | Alle Anzeigemodelle spielbarer Charaktere | Braucht deine AES-Schlüsseldatei; der Dateileser (aus `tools/wuwa_cli` gebaut) ist dabei; jeder Materialabschnitt (Haare, Gesicht, Oberkörper…) ist ein eigenes Teil |
| Vorhandene Konvertierungen | Fertige Modelle in `*_Converted`-Ordnern | Nach Prüfung des SHA-256 jeder Datei kopiert |

## Screenshots

Das englische Fenster, aufgenommen bei einem echten Scan und einer echten Vorschau mit `tools/screenshots.py` (Girls' Frontline 2).

| | |
|---|---|
| ![Scan einer Installation](../images/scan.png) | ![Ein Modell und seine Vorschau](../images/preview.png) |
| *Scan: Jeder zählbare Schritt hat seinen Balken* | *Die Modellliste und die Vorschau, auf das Modell eingepasst* |
| ![Der Modellbetrachter](../images/viewer.png) | ![Einstellungen](../images/settings.png) |
| *Der Modellbetrachter und der Inhalt der Datei* | *Einstellungen: Installationsordner, Werkzeuge, Ausgabe und Sprache* |

## Funktionen

- **Scannen, ankreuzen, exportieren.** Jedes Spiel hat eine Seite: Installation scannen, suchen, Modelle ankreuzen und in einer Warteschlange
  exportieren. Jeder Scan und jeder Export läuft in einem eigenen Prozess, ein fehlschlagender Konverter reißt das Fenster nie mit.
- **Findet deine Installationen.** Vor einem Scan bietet es die Kopien an, die es in deinen Steam-Bibliotheken findet; der gewählte Ordner kann der des
  Spiels sein, der darüber oder einer in seinen Daten.
- **Drei Formate, zwei ohne Blender.** `.glb` und FBX schreibt Polylode selbst (Skelett, geskinnte Meshes, Shape Keys, eingebettete Texturen,
  Transparenz); `.blend` braucht Blender. Ist Blender gesetzt, wird jede Datei zur Kontrolle darin zurückgelesen.
- **Materialien, wie die Spiele sie zeichnen.** Ausschnitte, Transparenz, Lidschatten-Ebenen und Augenglanz-Ebenen werden nach den Regeln geschrieben,
  die aus den Shadern jedes Spiels gelesen wurden. Wo glTF/FBX eine Ebene nicht mischen können, wird ihre Wirkung in die Textur darunter gebacken;
  die `.blend` behält echte Ebenen. Bei Wuthering Waves stammt die Ausschnitt-Abdeckung aus dem ausgeführten, kompilierten Shader des Spiels, und das Parallax-Fell wird als Haarschichten gebaut. Endfields Effektmaterialien (Flügel, Leuchten) laufen durch den kompilierten Shader des Spiels und werden gebacken, Emissionskarten als Emission geschrieben; ein nicht darstellbares Material wird unsichtbar gemacht und protokolliert, nie grau gezeichnet.
- **Vorschau vor dem Export.** Zeile wählen und Vorschau drücken (oder doppelklicken): Das Modell wird in einen Cache als `.glb` konvertiert und
  neben der Liste gezeigt — linke Maustaste dreht, rechte verschiebt, das Mausrad zoomt. **Convert again** erstellt sie neu, nachdem das Programm oder die Spieldateien aktualisiert wurden.
- **Modellbetrachter.** Listet alle Modelle im Ausgabeordner und öffnet jede `.glb`, `.gltf`, `.fbx` oder `.obj` (hineingezogen, gewählt oder als
  `Polylode.exe <Datei>`, damit es dein „Öffnen mit“-Programm sein kann): Vorder-, Rück-, Seiten- und Draufsicht, Drahtgitter, heller oder dunkler
  Hintergrund, Lichtstärke und der Inhalt eines glTF. Ein gelistetes Modell lässt sich in den Papierkorb löschen.
- **Über GitHub Pages in Discord teilen.** Discord zeigt keine 3D-Modelle; **Share on GitHub Pages…** im Betrachter
  schreibt die angezeigte `.glb` als Seite, die sie in jedem Browser dreht, in deine eigene GitHub-Pages-Website,
  pusht sie und kopiert den Link; in Discord eingefügt, erscheint der Link als Karte mit dem Bild des Modells.
- **Fünf Sprachen.** English, 繁體中文, 简体中文, 日本語 und 한국어; es folgt dem System, bis du in den Einstellungen eine wählst, und der Wechsel greift sofort.

## Download

Lade das neueste `Polylode-<Version>-windows-x64.zip` von [Releases](https://github.com/JE-Chen/Polylode/releases), entpacke es irgendwo und starte
`Polylode.exe`. Es bringt alles mit: AnimeStudio (für Endfield), den Wuthering-Waves-Leser, eine .NET-10-Laufzeit, den glTF-Importer und ein Software-OpenGL für Rechner ohne
GPU-Treiber.

Voraussetzungen: Windows 10 oder 11, 64 Bit. Blender 4.x oder 5.x nur für `.blend`.

## Anleitung

Das Fenster gibt es auf Englisch, Chinesisch, Japanisch und Koreanisch; diese Anleitung nennt die Schaltflächen so, wie das
englische Fenster sie zeigt.

### 1. Erster Start

Entpacke das Release irgendwo und starte `Polylode.exe`. Das Fenster folgt der Sprache von Windows; unter
**Settings → Language** wählst du eine andere, der Wechsel greift sofort. Einstellungen werden beim Tippen gespeichert und
liegen mit allen Caches in `%LOCALAPPDATA%\Polylode`. Hast du den Extraktor benutzt, als er Teil von IslandLife war,
werden beim ersten Start seine Einstellungen, die Endfield-Asset-Tabellen, der GF2-Index und die Speicher übernommen,
nichts wird neu gebaut.

### 2. Einstellungen

- **Ausgabeordner** — wohin Exporte gehen, ein Ordner pro Modell: `<Ausgabe>\<Spiel>\<Modell-ID>\`.
- **Formate** — `.glb`, FBX und `.blend`; kreuze an, was du brauchst.
- **Blender-Programm** — optional. Nur für `.blend` nötig; ist es gesetzt, wird jede `.glb` und FBX zur Kontrolle auch
  in Blender zurückgelesen.
- **Installationsordner** — dürfen leer bleiben: Ein Scan fragt nach jedem und bietet die Kopien aus deinen
  Steam-Bibliotheken an. Der Ordner darf der des Spiels sein, der darüber oder einer in seinen Daten. Ein ✓ markiert
  einen vorhandenen Pfad.
- **Werkzeuge** — leer lassen, um das Mitgelieferte zu nutzen: AnimeStudio (Endfield), den Wuthering-Waves-Leser und die
  .NET-Laufzeit. Wuthering Waves braucht außerdem deine **AES-Schlüsseldatei** (ein JSON mit `mainKey` und
  `dynamicKeys` für deine Client-Version): Die Dateien sind verschlüsselt, und der Schlüssel gehört dir, das Programm
  kann ihn nicht mitbringen. Vor der ersten Vorschau oder dem ersten Export sucht es eine Schlüsseldatei, die du schon hast — neben dem Programm, in seinem Datenordner, im Installationsordner und dem darüber, in Downloads, Dokumente und Desktop — und fragt nur, wenn es keine findet.

### 3. Scannen

Öffne die Seite eines Spiels und drücke **Scan install**. Der Balken unter der Statuszeile nennt den Schritt und zählt
ihn, wo es geht. Was ein erster Scan kostet:

- **Arknights: Endfield** baut die Asset-Tabellen aus der Installation: Dutzende Minuten, in Stapeln. Wird das Programm
  mittendrin geschlossen, geht nichts verloren; der nächste Scan macht weiter.
- **Girls' Frontline 2** indiziert die Bundles (Minuten) und prüft, welche Wurzeln Charaktermodelle sind.
- **Wuthering Waves** liest die Charaktertabellen aus den Archiven.

Spätere Scans verwenden alles weiter, bis das Spiel aktualisiert wird. Scanne nach einem Update erneut; eine
GF2-Vorschau oder ein GF2-Export meldet, wenn sich die Installation seit dem Scan geändert hat.

### 4. Vorschau

Wähle eine Zeile — die Suche filtert nach Charakter, Outfit oder ID — und drücke **Preview** oder doppelklicke. Die
erste Vorschau eines Modells konvertiert es (Sekunden bis eine Minute, mit eigenem Balken); danach zeigt die Auswahl
der Zeile es sofort aus dem Cache. In der Ansicht: **Ziehen mit links** dreht, **mit rechts** verschiebt, das
**Mausrad** zoomt. Jedes Modell wird beim Laden ganz und mittig eingepasst.

Eine Vorschau bleibt, wie sie zuerst erstellt wurde. **Convert again** konvertiert die gewählte Zeile neu — nach
einem Update von Polylode oder dem Spiel — und ersetzt die gespeicherte, sobald die neue fertig ist; bis dahin bleibt die alte.

### 5. Exportieren

Kreuze Modelle an (**Select all** und **Select none** wirken auf die Zeilen, die die Suche zeigt) und drücke
**Export ticked models**. Die Warteschlange unten arbeitet sie nacheinander ab: ⏳ wartet, ▶ läuft, ✓ geschrieben,
✗ fehlgeschlagen — das Protokoll daneben sagt warum. Der Balken unter der Warteschlange zählt Modelle; der darunter zeigt
den Schritt des gerade geschriebenen Modells. Wird das Fenster während eines Exports geschlossen, fragt es zuerst.

### 6. Modellbetrachter

Der Betrachter listet die Modelle im Ausgabeordner, die neuesten zuerst. Mit **Open file…**, einer auf das Fenster
gezogenen Datei oder `Polylode.exe <Datei>` (damit es dein „Öffnen mit“-Programm sein kann) siehst du jede `.glb`,
`.gltf`, `.fbx` oder `.obj`. **Front**, **Back**, **Left**, **Right** und **Top** drehen das Modell; **Wireframe**, die
Hintergrund-Schaltfläche und **Light** ändern die Darstellung. Die Leiste rechts listet, was ein glTF enthält — Meshes,
Vertices, Knochen, Shape Keys und die Mischung jedes Materials — und markiert eine nach alten Regeln geschriebene Datei,
die neu exportiert werden sollte. **Delete…** verschiebt das Modell, seine Geschwisterdateien und seinen Texturordner
in den Papierkorb.

### 7. Auf GitHub Pages teilen (für Discord)

Discord kann kein 3D-Modell anzeigen: Eine dort gepostete `.glb` ist nur ein Download. Polylode teilt ein Modell als eigene
Seite auf einer GitHub-Pages-Website in deinem eigenen GitHub-Konto; die Seite dreht das Modell in jedem Browser, und ihr Link
erscheint in Discord als Karte mit dem Bild des Modells. Sonst ist nichts zu installieren: kein Git, kein Klon.

1. Öffne eine `.glb` im Modellbetrachter und drücke **Share on GitHub Pages…**. Es nennt die Adresse, unter der das Modell
   öffentlich wird, und fragt vor dem Weitermachen.
2. Beim ersten Mal öffnet es GitHubs Anmeldeseite im Browser und zeigt einen kurzen Code (schon kopiert): Gib ihn ein und
   autorisiere Polylode. Es verlangt nur `public_repo` (öffentliche Repositorys anlegen und beschreiben) und bewahrt die
   Autorisierung nur auf diesem Computer auf, verschlüsselt für deinen Windows-Benutzer. **Settings → GitHub account → Sign out**
   vergisst sie; in GitHub widerruft Settings → Applications sie.
3. Es legt das öffentliche Repository `polylode-models` in deinem Konto an (einmal) und lädt `models/<name>/` (das Modell, ein
   Kartenbild 1200×630, gezeichnet wie die Vorschau, und die Seite) mit der Modellliste der Website als ein Commit hoch.
4. Der Link ist kopiert: `https://<konto>.github.io/polylode-models/models/<name>/`. Ist Pages für das Repository beim ersten
   Mal noch nicht an, bietet die Meldung **Open Pages settings**: wähle **Deploy from a branch**, `main`, `/ (root)` und Save,
   nur einmal. (Es aus dem Programm einzuschalten bräuchte Zugriff auf alle deine privaten Repositorys; das verlangt es nicht.)
5. Pages aktualisiert sich meist nach ein bis zwei Minuten; danach den Link in Discord einfügen.

**Posen.** Die exportierte `.glb` einer Endfield-Figur enthält die Posen ihrer Charakterbildschirme (die im Spiel
geloopten `ui`-Clips, beim ersten Mal aus der Installation dekodiert, etwa eine halbe Minute). Die geteilte Seite hat ein
Menü **Pose** zum Wechseln, und ihre Karte zeigt die erste. **Settings → Poses** schaltet das ab; Vorschauen enthalten sie
nie. Ein Outfit aus Girls' Frontline 2 enthält die
Ruheposen, in denen es abseits des Kampfes gezeigt wird (Kaserne, Garderobe, Schlafzimmer, Abspann: die geloopten Clips);
der erste Export benennt die Clips des Spiels einmal, ein paar Minuten. Figuren aus Wuthering Waves haben noch keine.

Schon eine eigene Pages-Website? Trage ihren lokalen Klon unter **Settings → GitHub Pages site folder** ein: Teilen committet
dann dort und pusht mit Git.

Nur `.glb` lässt sich teilen, bis 100 MB (GitHubs Dateigrenze); ein Modell erneut zu teilen ersetzt seine Seite. Alles
Geteilte ist öffentlich: Die Modelle stammen aus Spielen, also stelle sicher, dass du sie teilen darfst. Der Ordner des
Programms selbst wird als Website abgelehnt.

### Fehlerbehebung

- **Vor einem Scan, einer Vorschau oder einem Export öffnet sich eine Erklärung oder Dateiauswahl** — ein Werkzeug, ohne das die Konvertierung nicht läuft, ist nicht gesetzt und lag nicht dort, wo das Programm sucht. Bei Wuthering Waves ist es die AES-Schlüsseldatei: Das Fenster erklärt sie; wähle deine Schlüsseldatei oder füge den Schlüssel ein (`0x` und 64 Hexadezimalziffern), er wird als Schlüsseldatei gespeichert. So oder so bleibt es gemerkt.
- **„…has not been set“ oder „…is not there“** im Protokoll — korrigiere diesen Pfad unter **Settings**.
- **„This file's size could not be measured“** — der Betrachter konnte die Ausdehnung der Datei nicht lesen (etwa bei
  einer FBX 7.5); die Ansicht steht am Ursprung, zoome mit dem Mausrad.
- **Keine `.blend` geschrieben** — setze das Blender-Programm unter **Settings**.
- **„git push failed“ beim Teilen** — Git kann nicht in das Repository der Website pushen: Melde Git bei GitHub an (zum
  Beispiel mit dem Git Credential Manager) und prüfe, ob du aus diesem Ordner selbst pushen kannst. **„The site folder's
  origin is not a GitHub repository“** — der Ordner muss ein Klon eines GitHub-Repositorys sein.
- **Etwas anderes schlug fehl** — das Protokoll unten behält jede Zeile, die Scan oder Export ausgegeben haben.

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

Der mitgelieferte Wuthering-Waves-Leser wird von `tools/wuwa_cli/build.ps1` gebaut (siehe dessen `README.md`) und dem Build mit `--wuwa-reader` übergeben; die Workflows tun beides. `tools/screenshots.py` nimmt die Screenshots oben über Scan und Vorschau des Fensters selbst neu auf.

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
docs/                    Screenshots, Werkzeugnotizen, Architektur, offene Punkte, README-Übersetzungen
build.py, main.py        das gepackte Programm
```

## Lizenz und Inhalte

Der Code von Polylode steht unter der GNU LGPL 2.1 (`LICENSE`); das mitgelieferte AnimeStudio unter MIT (`tools/AnimeStudio/LICENSE`). Polylode
enthält keine Spielinhalte: Es konvertiert Dateien aus den Installationen auf deinem eigenen Rechner, und was es schreibt, bleibt dort. Die Screenshots zeigen ein Modell, das aus der eigenen Installation des Autors konvertiert wurde. Nutze es nur
mit Spielen, die du besitzt, und halte dich an ihre Bedingungen.
