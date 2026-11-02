# Build the Wuthering Waves reader into -Output, from a CUE4Parse checkout at the commit README.md names.
#
#   tools/wuwa_cli/build.ps1 -CUE4Parse tools/wuwa_cli/CUE4Parse -Output tools/WuwaReader
#
# Applies cue4parse_wuthering_waves.patch to the checkout (once: an applied patch is left as it is), builds
# wuwa.csproj in Release with the .NET 10 SDK on PATH, and puts CUE4Parse's licence beside the program.
# The CI and Release workflows run it on GitHub's runner; build.py copies -Output into the program.
[CmdletBinding(PositionalBinding = $false)]
param(
    [Parameter(Mandatory)] [string] $CUE4Parse,
    [Parameter(Mandatory)] [string] $Output
)
$ErrorActionPreference = 'Stop'
$here = $PSScriptRoot
$patch = Join-Path $here 'cue4parse_wuthering_waves.patch'
$CUE4Parse = (Resolve-Path $CUE4Parse).Path

git -C $CUE4Parse apply --reverse --check $patch 2>$null
if ($LASTEXITCODE -ne 0) {
    git -C $CUE4Parse apply $patch
    if ($LASTEXITCODE -ne 0) { throw "the patch does not apply to $CUE4Parse" }
}
dotnet build -c Release (Join-Path $here 'wuwa.csproj') "-p:CUE4ParseRoot=$CUE4Parse" -o $Output -nologo
if ($LASTEXITCODE -ne 0) { throw "dotnet build exit $LASTEXITCODE" }
# Detex is a resource inside CUE4Parse-Conversion that the reader would write beside itself on first use
# (DetexHelper.LoadDll); put there now, so a program folder the user cannot write still decodes textures
Copy-Item (Join-Path $CUE4Parse 'CUE4Parse-Conversion/Resources/Detex.dll') (Join-Path $Output 'Detex.dll')
foreach ($name in 'wuwa.exe', 'wuwa.dll', 'Detex.dll') {
    if (-not (Test-Path (Join-Path $Output $name))) { throw "the build wrote no $name" }
}
Copy-Item (Join-Path $CUE4Parse 'LICENSE') (Join-Path $Output 'CUE4Parse-LICENSE')
Get-ChildItem $Output -Filter *.pdb | Remove-Item
Write-Host "reader built: $Output"
