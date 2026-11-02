# Polylode

[English](../../README.md) · [繁體中文](README.zh-TW.md) · [简体中文](README.zh-CN.md) · [日本語](README.ja.md) · [한국어](README.ko.md) · [Español](README.es.md) · **Français** · [Deutsch](README.de.md)

Polylode extrait les modèles de personnages des jeux que vous avez installés — des *poly*gones tirés du *lode* (le filon) — et les écrit en
fichiers texturés et riggés qui s'ouvrent partout : glTF (`.glb`), FBX et Blender (`.blend`). C'est un programme de bureau Windows avec une
fenêtre, un aperçu qu'on tourne à la souris et une visionneuse de modèles.

| Jeu | Ce qu'il lit | Remarques |
|---|---|---|
| Arknights: Endfield | Tous les prefabs de personnages jouables | Les tables de ressources sont construites depuis votre installation à la première analyse (par lots vérifiés et reprenables), puis de nouveau après une mise à jour |
| Girls' Frontline 2: Exilium | Toutes les tenues des personnages jouables | Seules les racines avec leur propre maillage skinné et un squelette humain complet sont listées (ni animaux, ni accessoires, ni variantes vides) |
| Wuthering Waves | Tous les modèles d'affichage des personnages jouables | Nécessite votre fichier de clés AES ; le lecteur de fichiers (construit depuis `tools/wuwa_cli`) est fourni |
| Conversions existantes | Les modèles finis des dossiers `*_Converted` | Copiés après vérification du SHA-256 de chaque fichier |

## Fonctions

- **Analyser, cocher, exporter.** Chaque jeu a sa page : analysez l'installation, cherchez, cochez les modèles et exportez-les dans une file.
  Chaque analyse et chaque export tourne dans son propre processus : un convertisseur qui échoue n'emporte jamais la fenêtre.
- **Trouve vos installations.** Avant une analyse, il propose les copies trouvées dans vos bibliothèques Steam ; le dossier choisi peut être celui du
  jeu, celui du dessus ou un dossier dans ses données.
- **Trois formats, dont deux sans Blender.** `.glb` et FBX sont écrits par Polylode lui-même (squelette, maillages skinnés, shape keys, textures
  intégrées, transparence) ; `.blend` nécessite Blender. Si Blender est défini, chaque fichier y est relu pour vérification.
- **Matériaux tels que les jeux les dessinent.** Découpes, transparence, calques de fard à paupières et de reflets des yeux sont écrits selon les règles
  lues dans les shaders de chaque jeu. Là où glTF/FBX ne peuvent pas mélanger un calque, son effet est cuit dans la texture en dessous ; le `.blend`
  garde de vrais calques.
- **Aperçu avant export.** Sélectionnez une ligne et cliquez sur Aperçu (ou double-cliquez) : le modèle est converti en `.glb` dans un cache et
  affiché à côté de la liste — bouton gauche pour tourner, droit pour déplacer, molette pour zoomer.
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

## Utilisation

1. **Paramètres** — indiquez le dossier de sortie et les formats. Les dossiers d'installation peuvent rester vides : l'analyse les demande (et
   propose ceux trouvés dans Steam).
2. **Page d'un jeu** — *Analyser l'installation*. La première analyse d'Endfield construit ses tables de ressources (des dizaines de minutes,
   reprenable) ; la première de GF2 indexe les bundles. Les suivantes les réutilisent jusqu'à la prochaine mise à jour du jeu.
3. Cochez les modèles, *Exporter les modèles cochés*, et suivez la file et le journal en bas.
4. **Visionneuse de modèles** — regardez ce que vous avez exporté, comparez, supprimez.

Paramètres et caches se trouvent dans `%LOCALAPPDATA%\Polylode`. Si vous avez utilisé l'extracteur lorsqu'il faisait partie d'IslandLife, ses
paramètres, les tables de ressources d'Endfield, l'index de GF2 et les stockages sont déplacés au premier démarrage : rien n'est reconstruit.

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
build.py, main.py        le programme empaqueté
```

## Licence et contenu

Le code de Polylode est sous GNU LGPL 2.1 (`LICENSE`) ; AnimeStudio, qu'il embarque, est sous MIT (`tools/AnimeStudio/LICENSE`). Polylode ne contient
aucun contenu de jeu : il convertit les fichiers des installations de votre propre ordinateur, et ce qu'il écrit y reste. Ne l'utilisez qu'avec des
jeux que vous possédez et respectez leurs conditions.
