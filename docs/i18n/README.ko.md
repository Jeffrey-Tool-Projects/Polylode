# Polylode

[English](../../README.md) · [繁體中文](README.zh-TW.md) · [简体中文](README.zh-CN.md) · [日本語](README.ja.md) · **한국어** · [Español](README.es.md) · [Français](README.fr.md) · [Deutsch](README.de.md)

Polylode는 설치한 게임에서 캐릭터 모델을 캐내어——광맥(*lode*)에서 폴리곤(*poly*gon)을——텍스처와 리그가 있는, 어디서나 열 수 있는 파일로
기록합니다: glTF(`.glb`), FBX, Blender(`.blend`). 창, 마우스로 돌려 볼 수 있는 미리보기, 모델 뷰어를 갖춘 Windows 데스크톱 프로그램입니다.

| 게임 | 읽는 것 | 비고 |
|---|---|---|
| 명일방주: 엔드필드 | 모든 플레이어블 캐릭터의 프리팹 | 첫 스캔에서 설치로부터 자산 대응표를 만들고(나누어 검증, 중단 후 재개 가능) 게임 업데이트 후 다시 만듦 |
| 소녀전선 2: 망명 | 모든 플레이어블 캐릭터의 의상 | 자체 스킨 메시와 완전한 인간형 골격이 있는 루트만 나열(펫, 소품, 빈 변형 제외) |
| 명조 | 모든 플레이어블 캐릭터의 표시 모델 | AES 키 파일 필요(`tools/wuwa_cli`로 빌드한 게임 파일 리더는 포함됨) |
| 기존 변환물 | `*_Converted` 폴더의 완성 모델 | 파일마다 SHA-256을 확인한 뒤 복사 |

## 기능

- **스캔, 체크, 내보내기.** 게임마다 페이지가 있어 설치를 스캔하고, 검색하고, 모델을 체크해 대기열로 내보냅니다. 스캔과 내보내기는 각각 별도
  프로세스에서 실행되므로 변환기가 실패해도 창이 멈추지 않습니다.
- **설치를 자동으로 찾기.** 스캔 전에 Steam 라이브러리에서 찾은 설치를 후보로 보여 줍니다. 게임 폴더, 그 상위 폴더, 데이터 안쪽 중 아무것이나 고를 수 있습니다.
- **세 가지 형식, 그중 둘은 Blender 불필요.** `.glb`와 FBX는 Polylode가 직접 씁니다(골격, 스킨 메시, 셰이프 키, 내장 텍스처, 투명).
  `.blend`에는 Blender가 필요합니다. Blender를 지정하면 모든 파일을 Blender에서 다시 읽어 확인합니다.
- **게임 그대로의 재질.** 컷아웃, 반투명, 아이섀도 오버레이, 눈 하이라이트 층을 각 게임 셰이더에서 읽은 규칙대로 씁니다. glTF/FBX가 그릴 수 없는
  층은 그 효과를 아래 텍스처에 구워 넣고, `.blend`는 실제 층으로 남깁니다.
- **내보내기 전 미리보기.** 행을 고르고 「미리보기」(또는 더블클릭): 모델을 `.glb`로 변환해 캐시에 두고 목록 옆에 보여 줍니다——왼쪽 드래그 회전,
  오른쪽 드래그 이동, 휠 확대·축소.
- **모델 뷰어.** 출력 폴더의 모든 모델을 나열하고 어떤 `.glb`, `.gltf`, `.fbx`, `.obj`든 엽니다(끌어다 놓기, 선택, 또는 `Polylode.exe <파일>`.
  「연결 프로그램」으로 지정 가능): 정면·후면·측면·위쪽 시점, 와이어프레임, 밝은/어두운 배경, 조명 세기, glTF 내용. 목록의 모델은 휴지통으로 삭제할 수 있습니다.
- **다섯 가지 언어.** English, 繁體中文, 简体中文, 日本語, 한국어. 기본은 시스템을 따르며 「설정」에서 고르면 바로 바뀝니다.

## 다운로드

[Releases](https://github.com/JE-Chen/Polylode/releases)에서 최신 `Polylode-<버전>-windows-x64.zip`을 받아 아무 곳에나 풀고 `Polylode.exe`를
실행합니다. 필요한 것은 모두 들어 있습니다: AnimeStudio(엔드필드용), 명조 리더, .NET 10 런타임, glTF 가져오기 모듈, GPU 드라이버가 없는 컴퓨터용 소프트웨어 OpenGL.

요구 사항: Windows 10 또는 11, 64비트. `.blend`에만 Blender 4.x 또는 5.x가 필요합니다.

## 사용 방법

1. **설정**——출력 폴더와 내보내기 형식을 지정합니다. 설치 폴더는 비워 둬도 됩니다: 스캔할 때 묻습니다(Steam에서 찾은 것도 보여 줌).
2. **게임 페이지**——「설치 스캔」. 엔드필드 첫 스캔은 자산 대응표를 만듭니다(수십 분, 재개 가능). GF2 첫 스캔은 bundle 색인을 만듭니다.
   이후 스캔은 게임이 업데이트될 때까지 재사용합니다.
3. 모델을 체크하고 「체크한 모델 내보내기」를 누른 뒤 아래 대기열과 로그를 봅니다.
4. **모델 뷰어**——내보낸 결과를 보고, 비교하고, 삭제합니다.

설정과 캐시는 `%LOCALAPPDATA%\Polylode`에 있습니다. IslandLife의 일부였을 때 내보내기 도구를 썼다면, 첫 실행 때 그 설정, 엔드필드 자산 대응표,
GF2 색인과 저장소를 옮겨 오므로 다시 만들 필요가 없습니다.

## 소스에서 빌드

```powershell
python -m venv .venv
.venv\Scripts\python -m pip install -r requirements-dev.txt
.venv\Scripts\python -m polylode.extractor          # 소스에서 창 실행
.venv\Scripts\python -m pytest tests                # 테스트
.venv\Scripts\python build.py --dotnet <dotnet 폴더>   # dist\Polylode\Polylode.exe, 이어서 자체 점검
```

빌드는 Nuitka로 컴파일하고(C 컴파일러 필요: Visual Studio), AnimeStudio, .NET 10 런타임(`--dotnet`, 기본값 `DOTNET_ROOT`), 소프트웨어 OpenGL을
프로그램 옆에 두고 `Polylode.exe --selftest`를 실행합니다. 모든 변환 모듈, 미리보기 QML, 실제 `.glb`를 불러오며 하나라도 없으면 빌드가 실패합니다.

## CI/CD

- **CI**(`.github/workflows/ci.yml`)——`main`과 `dev`로의 푸시와 PR마다 Windows에서 린트, 테스트, 자체 점검을 실행합니다.
- **Release**(`.github/workflows/release.yml`)——`main`으로 푸시할 때마다 테스트, Nuitka 빌드와 패키지판 자체 점검을 거쳐 압축한 프로그램을 GitHub Release로 게시합니다.

개발은 `dev`, `main`이 릴리스 버전입니다.

## 구성

```
polylode/extractor/      창, 스캔, 작업, 미리보기, 뷰어, 언어
polylode/endfield/       명일방주: 엔드필드——자산 대응표, 변환, FBX 장면
polylode/gf2/            소녀전선 2——bundle, 플레이어블 명단, 변환, 눈 층 굽기
polylode/wuthering_waves/  명조——역할 표, 표시 모델 조립
polylode/interchange/    glTF와 FBX 작성기(Blender 불필요)
tools/                   Blender 스크립트, 아이콘, FBX SDK 확인, 명조 리더 소스, AnimeStudio
tests/                   pytest
build.py, main.py        패키지 프로그램
```

## 라이선스와 콘텐츠

Polylode 코드는 GNU LGPL 2.1(`LICENSE`), 포함된 AnimeStudio는 MIT(`tools/AnimeStudio/LICENSE`)입니다. Polylode에는 게임 콘텐츠가 들어 있지 않습니다:
여러분의 컴퓨터에 설치된 파일을 변환하며, 작성한 결과도 그 컴퓨터에 남습니다. 소유한 게임에만 사용하고 그 약관을 지키세요.
