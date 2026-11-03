# Polylode

[English](../../README.md) · [繁體中文](README.zh-TW.md) · [简体中文](README.zh-CN.md) · [日本語](README.ja.md) · [한국어](README.ko.md) · [Español](README.es.md) · **Français** · [Deutsch](README.de.md)

Polylode extrait les modèles de personnages des jeux que vous avez installés — des *poly*gones tirés du *lode* (le filon) — et les écrit en
fichiers texturés et riggés qui s'ouvrent partout : glTF (`.glb`), FBX et Blender (`.blend`). C'est un programme de bureau Windows avec une
fenêtre, un aperçu qu'on tourne à la souris et une visionneuse de modèles.

| Jeu | Ce qu'il lit | Remarques |
|---|---|---|
| Arknights: Endfield | Tous les prefabs de personnages jouables | Les tables de ressources sont construites depuis votre installation à la première analyse (par lots vérifiés et reprenables), puis de nouveau après une mise à jour |
| Girls' Frontline 2: Exilium | Toutes les tenues des personnages jouables | Seules les racines avec leur propre maillage skinné et un squelette humain complet sont listées (ni animaux, ni accessoires, ni variantes vides) |
| Wuthering Waves | Tous les modèles d'affichage des personnages jouables | Nécessite votre fichier de clés AES ; le lecteur de fichiers (construit depuis `tools/wuwa_cli`) est fourni ; chaque section de matériau (cheveux, visage, buste…) est une pièce à part |
| Conversions existantes | Les modèles finis des dossiers `*_Converted` | Copiés après vérification du SHA-256 de chaque fichier |

## Captures d'écran

La fenêtre en anglais, prise pendant une vraie analyse et un vrai aperçu avec `tools/screenshots.py` (Girls' Frontline 2).

| | |
|---|---|
| ![Analyse d'une installation](../images/scan.png) | ![Un modèle et son aperçu](../images/preview.png) |
| *Analyse : chaque étape qui se compte a sa barre* | *La liste des modèles et l'aperçu, cadré sur le modèle* |
| ![La visionneuse](../images/viewer.png) | ![Paramètres](../images/settings.png) |
| *La visionneuse et le contenu du fichier* | *Paramètres : dossiers d'installation, outils, sortie et langue* |

## Fonctions

- **Analyser, cocher, exporter.** Chaque jeu a sa page : analysez l'installation, cherchez, cochez les modèles et exportez-les dans une file.
  Chaque analyse et chaque export tourne dans son propre processus : un convertisseur qui échoue n'emporte jamais la fenêtre.
- **Trouve vos installations.** Avant une analyse, il propose les copies trouvées dans vos bibliothèques Steam ; le dossier choisi peut être celui du
  jeu, celui du dessus ou un dossier dans ses données.
- **Trois formats, dont deux sans Blender.** `.glb` et FBX sont écrits par Polylode lui-même (squelette, maillages skinnés, shape keys, textures
  intégrées, transparence) ; `.blend` nécessite Blender. Si Blender est défini, chaque fichier y est relu pour vérification.
- **Matériaux tels que les jeux les dessinent.** Découpes, transparence, calques de fard à paupières et de reflets des yeux sont écrits selon les règles
  lues dans les shaders de chaque jeu. Là où glTF/FBX ne peuvent pas mélanger un calque, son effet est cuit dans la texture en dessous ; le `.blend`
  garde de vrais calques. Pour Wuthering Waves, la couverture des découpes vient de l'exécution du shader compilé du jeu, et la fourrure en parallaxe est construite en couches de mèches.
- **Aperçu avant export.** Sélectionnez une ligne et cliquez sur Aperçu (ou double-cliquez) : le modèle est converti en `.glb` dans un cache et
  affiché à côté de la liste — bouton gauche pour tourner, droit pour déplacer, molette pour zoomer. **Convert again** le refait après une mise à jour du programme ou des fichiers du jeu.
- **Visionneuse de modèles.** Liste tous les modèles du dossier de sortie et ouvre tout `.glb`, `.gltf`, `.fbx` ou `.obj` (déposé, choisi, ou passé
  en `Polylode.exe <fichier>` pour servir de programme « Ouvrir avec ») : vues de face, de dos, de côté et de dessus, fil de fer, fond clair ou
  sombre, intensité de la lumière et contenu d'un glTF. Un modèle listé peut être envoyé à la corbeille.
- **Cinq langues.** English, 繁體中文, 简体中文, 日本語 et 한국어 ; il suit le système jusqu'à ce que vous en choisissiez une dans Paramètres, et le changement
  est immédiat.

## Téléchargement

Prenez le dernier `Polylode-<version>-windows-x64.zip` sur la page [Releases](https://github.com/JE-Chen/Polylode/releases), décompressez-le où vous
voulez et lancez `Polylode.exe`. Il contient tout ce qu'il lui faut : AnimeStudio (pour Endfield), le lecteur de Wuthering Waves, un runtime .NET 10, l'importateur glTF et un
OpenGL logiciel pour les ordinateurs sans pilote de GPU.

Configuration : Windows 10 ou 11, 64 bits. Blender 4.x ou 5.x uniquement pour `.blend`.

## Guide d'utilisation

La fenêtre existe en anglais, chinois, japonais et coréen ; ce guide nomme ses boutons comme la fenêtre anglaise les affiche.

### 1. Premier démarrage

Décompressez la release où vous voulez et lancez `Polylode.exe`. La fenêtre suit la langue de Windows ; choisissez-en
une autre dans **Settings → Language**, le changement est immédiat. Les paramètres sont enregistrés à la saisie et se
trouvent, avec tous les caches, dans `%LOCALAPPDATA%\Polylode`. Si vous avez utilisé l'extracteur lorsqu'il faisait
partie d'IslandLife, ses paramètres, les tables de ressources d'Endfield, l'index de GF2 et les stockages sont déplacés
au premier démarrage : rien n'est reconstruit.

### 2. Paramètres

- **Dossier de sortie** — où vont les exports, un dossier par modèle : `<sortie>\<jeu>\<id du modèle>\`.
- **Formats** — `.glb`, FBX et `.blend` ; cochez ceux que vous voulez.
- **Exécutable Blender** — facultatif. Nécessaire uniquement pour `.blend` ; s'il est défini, chaque `.glb` et FBX est
  aussi relu dans Blender pour vérification.
- **Dossiers d'installation** — peuvent rester vides : l'analyse demande chacun et propose les copies trouvées dans vos
  bibliothèques Steam. Le dossier peut être celui du jeu, celui du dessus ou un dossier dans ses données. Un ✓ marque un
  chemin qui existe.
- **Outils** — laissez-les vides pour utiliser ce que le programme embarque : AnimeStudio (Endfield), le lecteur de
  Wuthering Waves et le runtime .NET. Wuthering Waves demande aussi votre **fichier de clés AES** (un JSON avec
  `mainKey` et `dynamicKeys` pour votre version du client) : ses fichiers sont chiffrés, et la clé est la vôtre, pas
  quelque chose que le programme peut embarquer. Avant le premier aperçu ou export, il cherche un fichier de clés que vous avez déjà — à côté du programme, dans son dossier de données, dans le dossier d'installation et celui du dessus, dans Téléchargements, Documents et Bureau — et ne le demande que s'il n'en trouve aucun.

### 3. Analyser

Ouvrez la page d'un jeu et cliquez sur **Scan install**. La barre sous la ligne d'état nomme l'étape et la compte quand
c'est possible. Ce que coûte une première analyse :

- **Arknights: Endfield** construit ses tables de ressources depuis l'installation : des dizaines de minutes, par lots.
  Fermer le programme en cours de route ne perd rien ; l'analyse suivante reprend.
- **Girls' Frontline 2** indexe les bundles (quelques minutes) et vérifie quelles racines sont des modèles de personnage.
- **Wuthering Waves** lit ses tables de personnages dans les archives.

Les analyses suivantes réutilisent tout jusqu'à la mise à jour du jeu. Relancez une analyse après une mise à jour ; un
aperçu ou un export GF2 vous prévient si l'installation a changé depuis son analyse.

### 4. Aperçu

Choisissez une ligne — la recherche filtre par personnage, tenue ou id — et cliquez sur **Preview** ou double-cliquez.
Le premier aperçu d'un modèle le convertit (de quelques secondes à une minute, avec sa propre barre) ; ensuite,
sélectionner la ligne l'affiche aussitôt depuis le cache. Dans la vue : **glisser avec le bouton gauche** fait tourner,
**avec le droit** déplace, la **molette** zoome. Chaque modèle est cadré entier et centré au chargement.

Un aperçu reste tel qu'il a été fait la première fois. **Convert again** convertit à nouveau la ligne choisie — après une
mise à jour de Polylode ou du jeu — et remplace celui du cache une fois le nouveau prêt ; d'ici là, l'ancien est gardé.

### 5. Exporter

Cochez les modèles (**Select all** et **Select none** agissent sur les lignes affichées par la recherche) et cliquez sur
**Export ticked models**. La file en bas les traite un par un : ⏳ en attente, ▶ en cours, ✓ écrit, ✗ échec — le journal
à côté dit pourquoi. La barre sous la file compte les modèles ; celle d'en dessous montre l'étape du modèle en cours
d'écriture. Fermer la fenêtre pendant un export demande d'abord confirmation.

### 6. Visionneuse

La visionneuse liste les modèles du dossier de sortie, les plus récents d'abord. **Open file…**, un fichier déposé sur
la fenêtre ou `Polylode.exe <fichier>` (pour servir de programme « Ouvrir avec ») affichent tout `.glb`, `.gltf`,
`.fbx` ou `.obj`. **Front**, **Back**, **Left**, **Right** et **Top** tournent le modèle ; **Wireframe**, le bouton de
fond et **Light** changent son rendu. Le panneau de droite liste ce que contient un glTF — maillages, sommets, os, shape
keys et le mélange de chaque matériau — et signale un fichier écrit selon d'anciennes règles, à exporter de nouveau.
**Delete…** envoie le modèle, ses fichiers frères et son dossier de textures à la corbeille.

### Dépannage

- **Une explication ou un sélecteur de fichier s'ouvre avant une analyse, un aperçu ou un export** — un outil indispensable à la conversion n'est pas défini et n'était pas là où le programme cherche. Pour Wuthering Waves, c'est le fichier de clés AES : la fenêtre explique ce que c'est ; choisissez votre fichier de clés ou collez la clé (`0x` et 64 chiffres hexadécimaux), elle est enregistrée comme fichier de clés. Dans les deux cas, c'est retenu.
- **« …has not been set » ou « …is not there »** dans le journal — corrigez ce chemin dans **Settings**.
- **« This file's size could not be measured »** — la visionneuse n'a pas pu lire l'étendue du fichier (un FBX 7.5, par
  exemple) ; la vue est à l'origine, zoomez avec la molette.
- **Aucun `.blend` écrit** — indiquez l'exécutable Blender dans **Settings**.
- **Autre échec** — le journal en bas garde chaque ligne affichée par l'analyse ou l'export.

## Compiler depuis les sources

```powershell
python -m venv .venv
.venv\Scripts\python -m pip install -r requirements-dev.txt
.venv\Scripts\python -m polylode.extractor          # la fenêtre depuis les sources
.venv\Scripts\python -m pytest tests                # tests
.venv\Scripts\python build.py --dotnet <dossier dotnet>   # dist\Polylode\Polylode.exe, puis son autotest
```

La compilation passe par Nuitka (un compilateur C est nécessaire : celui de Visual Studio), copie à côté du programme AnimeStudio, un runtime .NET 10
(`--dotnet`, par défaut `DOTNET_ROOT`) et l'OpenGL logiciel, puis lance `Polylode.exe --selftest`, qui charge chaque convertisseur, le QML de
l'aperçu et un vrai `.glb` ; s'il manque quoi que ce soit, la compilation échoue.

Le lecteur de Wuthering Waves embarqué est compilé par `tools/wuwa_cli/build.ps1` (voir son `README.md`) et passé à la compilation avec `--wuwa-reader` ; les workflows font les deux. `tools/screenshots.py` reprend les captures ci-dessus avec l'analyse et l'aperçu de la fenêtre elle-même.

## CI/CD

- **CI** (`.github/workflows/ci.yml`) — à chaque push et pull request vers `main` et `dev` : lint, tests et autotest sous Windows.
- **Release** (`.github/workflows/release.yml`) — à chaque push vers `main` : tests, compilation Nuitka avec l'autotest du programme empaqueté, et une
  GitHub Release avec le programme compressé.

Le travail se fait sur `dev` ; `main` est ce qui est publié.

## Organisation

```
polylode/extractor/      la fenêtre, les analyses, les tâches, l'aperçu, la visionneuse, les langues
polylode/endfield/       Arknights: Endfield — tables de ressources, conversion, scène FBX
polylode/gf2/            Girls' Frontline 2 — bundles, liste jouable, conversion, cuisson des calques des yeux
polylode/wuthering_waves/  Wuthering Waves — tables des rôles, assemblage des modèles d'affichage
polylode/interchange/    les écrivains glTF et FBX (sans Blender)
tools/                   scripts Blender, l'icône, la vérification par le SDK FBX, les sources du lecteur WuWa, AnimeStudio
tests/                   pytest
docs/                    captures, notes sur les outils, architecture, points ouverts, traductions du README
build.py, main.py        le programme empaqueté
```

## Licence et contenu

Le code de Polylode est sous GNU LGPL 2.1 (`LICENSE`) ; AnimeStudio, qu'il embarque, est sous MIT (`tools/AnimeStudio/LICENSE`). Polylode ne contient
aucun contenu de jeu : il convertit les fichiers des installations de votre propre ordinateur, et ce qu'il écrit y reste. Les captures montrent un modèle converti depuis l'installation de l'auteur. Ne l'utilisez qu'avec des
jeux que vous possédez et respectez leurs conditions.
