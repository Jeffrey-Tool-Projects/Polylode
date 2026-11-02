# Polylode

[English](../../README.md) · [繁體中文](README.zh-TW.md) · [简体中文](README.zh-CN.md) · [日本語](README.ja.md) · [한국어](README.ko.md) · **Español** · [Français](README.fr.md) · [Deutsch](README.de.md)

Polylode extrae los modelos de personajes de los juegos que tienes instalados —*poly*gonos de la *lode* (veta)— y los
escribe como archivos con texturas y esqueleto que se abren en cualquier parte: glTF (`.glb`), FBX y Blender (`.blend`).
Es un programa de escritorio para Windows con ventana, una vista previa que se gira con el ratón y un visor de modelos.

| Juego | Qué lee | Notas |
|---|---|---|
| Arknights: Endfield | Todos los prefabs de personajes jugables | Los mapas de recursos se construyen desde tu instalación en el primer escaneo (por lotes comprobados y reanudables) y de nuevo tras una actualización |
| Girls' Frontline 2: Exilium | Todos los atuendos de personajes jugables | Solo raíces con su propia malla con skin y un esqueleto humano completo (sin mascotas, accesorios ni variantes vacías) |
| Wuthering Waves | Todos los modelos de presentación de personajes jugables | Necesita el lector de archivos construido desde `tools/wuwa_cli` y tu archivo de claves AES |
| Conversiones existentes | Modelos terminados en carpetas `*_Converted` | Se copian tras comprobar el SHA-256 de cada archivo |

## Funciones

- **Escanear, marcar, exportar.** Cada juego tiene su página: escanea la instalación, busca, marca los modelos y expórtalos en una cola.
  Cada escaneo y exportación corre en su propio proceso, así que un conversor que falla nunca tumba la ventana.
- **Encuentra tus instalaciones.** Antes de escanear ofrece las copias que encuentra en tus bibliotecas de Steam; la carpeta elegida puede ser la del
  juego, la de encima o una dentro de sus datos.
- **Tres formatos, dos sin Blender.** `.glb` y FBX los escribe Polylode (esqueleto, mallas con skin, shape keys, texturas incrustadas,
  transparencia); `.blend` necesita Blender. Con Blender configurado, cada archivo se vuelve a leer en él como comprobación.
- **Materiales como los dibujan los juegos.** Recortes, transparencia, capas de sombra de ojos y de brillo ocular se escriben con las reglas leídas de los
  shaders de cada juego. Donde glTF/FBX no pueden mezclar una capa, su efecto se hornea en la textura de debajo; el `.blend` conserva las capas reales.
- **Vista previa antes de exportar.** Selecciona una fila y pulsa Vista previa (o doble clic): el modelo se convierte a `.glb` en una caché y se
  muestra junto a la lista; botón izquierdo para girar, derecho para desplazar, rueda para acercar.
- **Visor de modelos.** Lista todos los modelos de tu carpeta de salida y abre cualquier `.glb`, `.gltf`, `.fbx` u `.obj` (arrastrado, elegido o
  pasado como `Polylode.exe <archivo>` para usarlo en «Abrir con»): vistas frontal, trasera, laterales y superior, malla de alambre, fondo claro u
  oscuro, intensidad de la luz y el contenido de un glTF. Un modelo de la lista puede enviarse a la papelera.
- **Cinco idiomas.** English, 繁體中文, 简体中文, 日本語 y 한국어; sigue el idioma del sistema hasta que elijas uno en Ajustes, y el cambio es inmediato.

## Descarga

Descarga el último `Polylode-<versión>-windows-x64.zip` de [Releases](https://github.com/JE-Chen/Polylode/releases), descomprímelo donde quieras y
ejecuta `Polylode.exe`. Lleva todo lo que necesita: AnimeStudio (para Endfield), un runtime de .NET 10, el importador de glTF y un OpenGL por
software para equipos sin controlador de GPU.

Requisitos: Windows 10 u 11 de 64 bits. Blender 4.x o 5.x solo para `.blend`.

## Uso

1. **Ajustes** — indica la carpeta de salida y los formatos. Las carpetas de instalación pueden quedar vacías: el escaneo las pide (y ofrece las de Steam).
2. **Página de un juego** — *Escanear instalación*. El primer escaneo de Endfield construye sus mapas de recursos (decenas de minutos, reanudable);
   el primero de GF2 indexa los bundles. Los siguientes reutilizan ambos hasta que el juego se actualice.
3. Marca los modelos, pulsa *Exportar los marcados* y sigue la cola y el registro abajo.
4. **Visor de modelos** — mira lo exportado, compáralo, bórralo.

Los ajustes y cachés están en `%LOCALAPPDATA%\Polylode`. Si usaste el extractor cuando formaba parte de IslandLife, sus ajustes, mapas de recursos de
Endfield, índice de GF2 y almacenes se trasladan en el primer arranque, así que nada se vuelve a construir.

## Compilar desde el código

```powershell
python -m venv .venv
.venv\Scripts\python -m pip install -r requirements-dev.txt
.venv\Scripts\python -m polylode.extractor          # la ventana desde el código
.venv\Scripts\python -m pytest tests                # pruebas
.venv\Scripts\python build.py --dotnet <carpeta de dotnet>   # dist\Polylode\Polylode.exe y su autocomprobación
```

La compilación usa Nuitka (necesita un compilador de C: el de Visual Studio), copia junto al programa AnimeStudio, un runtime de .NET 10
(`--dotnet`, por defecto `DOTNET_ROOT`) y el OpenGL por software, y ejecuta `Polylode.exe --selftest`, que carga todos los conversores, el QML de
la vista previa y un `.glb` real; si falta algo, la compilación falla.

## CI/CD

- **CI** (`.github/workflows/ci.yml`) — en cada push y pull request a `main` y `dev`: lint, pruebas y autocomprobación en Windows.
- **Release** (`.github/workflows/release.yml`) — en cada push a `main`: pruebas, compilación con Nuitka con la autocomprobación del programa
  empaquetado y un GitHub Release con el programa comprimido.

El trabajo va a `dev`; `main` es lo que se publica.

## Estructura

```
polylode/extractor/      la ventana, escaneos, trabajos, vista previa, visor, idiomas
polylode/endfield/       Arknights: Endfield — mapas de recursos, conversión, escena FBX
polylode/gf2/            Girls' Frontline 2 — bundles, lista jugable, conversión, horneado de capas oculares
polylode/wuthering_waves/  Wuthering Waves — tablas de roles, ensamblado de modelos de presentación
polylode/interchange/    los escritores de glTF y FBX (sin Blender)
tools/                   scripts de Blender, el icono, la comprobación con el SDK de FBX, el código del lector de WuWa, AnimeStudio
tests/                   pytest
build.py, main.py        el programa empaquetado
```

## Licencia y contenido

El código de Polylode está bajo la GNU LGPL 2.1 (`LICENSE`); AnimeStudio, que incluye, es MIT (`tools/AnimeStudio/LICENSE`). Polylode no contiene
contenido de ningún juego: convierte archivos de las instalaciones de tu propio equipo, y lo que escribe se queda ahí. Úsalo solo con juegos que
poseas y respeta sus condiciones.
