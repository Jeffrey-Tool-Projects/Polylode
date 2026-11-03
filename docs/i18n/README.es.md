# Polylode

[English](../../README.md) · [繁體中文](README.zh-TW.md) · [简体中文](README.zh-CN.md) · [日本語](README.ja.md) · [한국어](README.ko.md) · **Español** · [Français](README.fr.md) · [Deutsch](README.de.md)

Polylode extrae los modelos de personajes de los juegos que tienes instalados —*poly*gonos de la *lode* (veta)— y los
escribe como archivos con texturas y esqueleto que se abren en cualquier parte: glTF (`.glb`), FBX y Blender (`.blend`).
Es un programa de escritorio para Windows con ventana, una vista previa que se gira con el ratón y un visor de modelos.

| Juego | Qué lee | Notas |
|---|---|---|
| Arknights: Endfield | Todos los prefabs de personajes jugables | Los mapas de recursos se construyen desde tu instalación en el primer escaneo (por lotes comprobados y reanudables) y de nuevo tras una actualización |
| Girls' Frontline 2: Exilium | Todos los atuendos de personajes jugables | Solo raíces con su propia malla con skin y un esqueleto humano completo (sin mascotas, accesorios ni variantes vacías). Un atuendo con piezas intercambiables se muestra como el juego lo viste por defecto; las otras versiones van ocultas en el archivo |
| Wuthering Waves | Todos los modelos de presentación de personajes jugables | Necesita tu archivo de claves AES; el lector de archivos (construido desde `tools/wuwa_cli`) viene incluido; cada sección de material (pelo, cara, torso…) es una pieza propia |
| Conversiones existentes | Modelos terminados en carpetas `*_Converted` | Se copian tras comprobar el SHA-256 de cada archivo |

## Capturas

La ventana en inglés, tomada de un escaneo y una vista previa reales con `tools/screenshots.py` (se ve Girls' Frontline 2).

| | |
|---|---|
| ![Escaneo de una instalación](../images/scan.png) | ![Un modelo y su vista previa](../images/preview.png) |
| *Escaneo: cada paso que se puede contar tiene su barra* | *La lista de modelos y su vista previa, encuadrada al modelo* |
| ![El visor de modelos](../images/viewer.png) | ![Ajustes](../images/settings.png) |
| *El visor de modelos y lo que contiene el archivo* | *Ajustes: carpetas de instalación, herramientas, salida e idioma* |

## Funciones

- **Escanear, marcar, exportar.** Cada juego tiene su página: escanea la instalación, busca, marca los modelos y expórtalos en una cola.
  Cada escaneo y exportación corre en su propio proceso, así que un conversor que falla nunca tumba la ventana.
- **Encuentra tus instalaciones.** Antes de escanear ofrece las copias que encuentra en tus bibliotecas de Steam; la carpeta elegida puede ser la del
  juego, la de encima o una dentro de sus datos.
- **Tres formatos, dos sin Blender.** `.glb` y FBX los escribe Polylode (esqueleto, mallas con skin, shape keys, texturas incrustadas,
  transparencia); `.blend` necesita Blender. Con Blender configurado, cada archivo se vuelve a leer en él como comprobación.
- **Materiales como los dibujan los juegos.** Recortes, transparencia, capas de sombra de ojos y de brillo ocular se escriben con las reglas leídas de los
  shaders de cada juego. Donde glTF/FBX no pueden mezclar una capa, su efecto se hornea en la textura de debajo; el `.blend` conserva las capas reales. En Wuthering Waves, la cobertura de los recortes sale de ejecutar el shader compilado del propio juego, y el pelaje con paralaje se construye como capas de mechones.
- **Vista previa antes de exportar.** Selecciona una fila y pulsa Vista previa (o doble clic): el modelo se convierte a `.glb` en una caché y se
  muestra junto a la lista; botón izquierdo para girar, derecho para desplazar, rueda para acercar. **Convert again** la rehace tras actualizar el programa o los archivos del juego.
- **Visor de modelos.** Lista todos los modelos de tu carpeta de salida y abre cualquier `.glb`, `.gltf`, `.fbx` u `.obj` (arrastrado, elegido o
  pasado como `Polylode.exe <archivo>` para usarlo en «Abrir con»): vistas frontal, trasera, laterales y superior, malla de alambre, fondo claro u
  oscuro, intensidad de la luz y el contenido de un glTF. Un modelo de la lista puede enviarse a la papelera.
- **Cinco idiomas.** English, 繁體中文, 简体中文, 日本語 y 한국어; sigue el idioma del sistema hasta que elijas uno en Ajustes, y el cambio es inmediato.

## Descarga

Descarga el último `Polylode-<versión>-windows-x64.zip` de [Releases](https://github.com/JE-Chen/Polylode/releases), descomprímelo donde quieras y
ejecuta `Polylode.exe`. Lleva todo lo que necesita: AnimeStudio (para Endfield), el lector de Wuthering Waves, un runtime de .NET 10, el importador de glTF y un OpenGL por
software para equipos sin controlador de GPU.

Requisitos: Windows 10 u 11 de 64 bits. Blender 4.x o 5.x solo para `.blend`.

## Guía de uso

La ventana está en inglés, chino, japonés y coreano; esta guía nombra los botones como los muestra la ventana en inglés.

### 1. Primer arranque

Descomprime la release donde quieras y ejecuta `Polylode.exe`. La ventana sigue el idioma de Windows; elige otro en
**Settings → Language** y cambia al momento. Los ajustes se guardan mientras escribes y viven, con todas las cachés, en
`%LOCALAPPDATA%\Polylode`. Si usaste el extractor cuando formaba parte de IslandLife, sus ajustes, mapas de recursos de
Endfield, índice de GF2 y almacenes se trasladan en el primer arranque, así que nada se vuelve a construir.

### 2. Ajustes

- **Carpeta de salida** — adónde van las exportaciones, una carpeta por modelo: `<salida>\<juego>\<id del modelo>\`.
- **Formatos** — `.glb`, FBX y `.blend`; marca los que quieras.
- **Ejecutable de Blender** — opcional. Solo hace falta para `.blend`; si está puesto, cada `.glb` y FBX también se
  vuelve a leer en Blender como comprobación.
- **Carpetas de instalación** — pueden quedar vacías: el escaneo pregunta por cada una y ofrece las copias que encuentra
  en tus bibliotecas de Steam. Vale la carpeta del juego, la de encima o una dentro de sus datos. Un ✓ marca una ruta que existe.
- **Herramientas** — déjalas vacías para usar lo que trae el programa: AnimeStudio (Endfield), el lector de Wuthering
  Waves y el runtime de .NET. Wuthering Waves necesita además tu **archivo de claves AES** (un JSON con `mainKey` y
  `dynamicKeys` de tu versión del cliente): sus archivos están cifrados y la clave es tuya, no algo que el programa
  pueda incluir. Antes de la primera vista previa o exportación busca un archivo de claves que ya tengas —junto al programa, en su carpeta de datos, en la carpeta de instalación y la de encima, en Descargas, Documentos y Escritorio— y solo lo pide si no encuentra ninguno.

### 3. Escanear

Abre la página de un juego y pulsa **Scan install**. La barra bajo la línea de estado nombra el paso y lo cuenta cuando
puede. Lo que cuesta un primer escaneo:

- **Arknights: Endfield** construye sus mapas de recursos desde la instalación: decenas de minutos, por lotes. Cerrar el
  programa a medias no pierde nada; el siguiente escaneo continúa.
- **Girls' Frontline 2** indexa los bundles (minutos) y comprueba qué raíces son modelos de personaje.
- **Wuthering Waves** lee sus tablas de personajes de los archivos.

Los escaneos siguientes lo reutilizan todo hasta que el juego se actualice. Vuelve a escanear tras una actualización;
una vista previa o exportación de GF2 te avisa si la instalación cambió desde su escaneo.

### 4. Vista previa

Elige una fila —el buscador filtra por personaje, atuendo o id— y pulsa **Preview** o haz doble clic. La primera
vista previa de un modelo lo convierte (de segundos a un minuto, con su propia barra); después, seleccionar la fila lo
muestra desde la caché al instante. En la vista: **arrastrar con el izquierdo** gira, **con el derecho** desplaza y la
**rueda** acerca. Cada modelo se encuadra entero y centrado al cargarse.

Una vista previa se queda como se hizo la primera vez. **Convert again** vuelve a convertir la fila elegida —tras
actualizar Polylode o el juego— y sustituye la guardada cuando la nueva termina; hasta entonces se conserva la anterior.

### 5. Exportar

Marca los modelos (**Select all** y **Select none** actúan sobre las filas que muestra la búsqueda) y pulsa
**Export ticked models**. La cola de abajo los hace de uno en uno: ⏳ en espera, ▶ en curso, ✓ escrito, ✗ falló —el
registro de al lado dice por qué. La barra bajo la cola cuenta modelos; la de debajo muestra el paso del modelo que se
está escribiendo. Cerrar la ventana durante una exportación pregunta antes.

### 6. Visor de modelos

El visor lista los modelos de tu carpeta de salida, los más nuevos primero. Con **Open file…**, soltando un archivo en
la ventana o iniciando `Polylode.exe <archivo>` (para usarlo en «Abrir con») puedes ver cualquier `.glb`, `.gltf`,
`.fbx` u `.obj`. **Front**, **Back**, **Left**, **Right** y **Top** giran el modelo; **Wireframe**, el botón de fondo y
**Light** cambian cómo se dibuja. El panel derecho lista lo que contiene un glTF —mallas, vértices, huesos, shape keys
y la mezcla de cada material— y marca un archivo escrito con reglas antiguas, que conviene exportar de nuevo.
**Delete…** envía el modelo, sus archivos hermanos y su carpeta de texturas a la papelera.

### Solución de problemas

- **Se abre un aviso o un selector de archivos antes de un escaneo, vista previa o exportación** — falta una herramienta sin la que la conversión no puede funcionar y no estaba donde busca el programa. En Wuthering Waves es el archivo de claves AES: el aviso explica qué es; elige tu archivo de claves o pega la clave (`0x` y 64 dígitos hexadecimales) y se guarda como archivo de claves. En ambos casos se recuerda.
- **«…has not been set» o «…is not there»** en el registro — corrige esa ruta en **Settings**.
- **«This file's size could not be measured»** — el visor no pudo leer la extensión del archivo (por ejemplo, un FBX
  7.5); la vista está en el origen, acerca con la rueda.
- **No se escribe el `.blend`** — indica el ejecutable de Blender en **Settings**.
- **Otro fallo** — el registro de abajo guarda cada línea que imprimió el escaneo o la exportación.

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

El lector de Wuthering Waves que trae el programa se compila con `tools/wuwa_cli/build.ps1` (ver su `README.md`) y se pasa a la compilación con `--wuwa-reader`; los flujos de trabajo hacen ambas cosas. `tools/screenshots.py` vuelve a tomar las capturas de arriba con el escaneo y la vista previa de la propia ventana.

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
docs/                    capturas, notas de herramientas, arquitectura, pendientes, traducciones del README
build.py, main.py        el programa empaquetado
```

## Licencia y contenido

El código de Polylode está bajo la GNU LGPL 2.1 (`LICENSE`); AnimeStudio, que incluye, es MIT (`tools/AnimeStudio/LICENSE`). Polylode no contiene
contenido de ningún juego: convierte archivos de las instalaciones de tu propio equipo, y lo que escribe se queda ahí. Las capturas muestran un modelo convertido desde la instalación del autor. Úsalo solo con juegos que
poseas y respeta sus condiciones.
