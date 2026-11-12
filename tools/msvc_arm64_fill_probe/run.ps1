# Build and run probe.cpp with every installed ARM64 MSVC toolset, at /Ox (Warp's
# release flag) and /O2. Assembly listings land in probe-out/ for upload.

$ErrorActionPreference = 'Stop'

$vs = & "${env:ProgramFiles(x86)}\Microsoft Visual Studio\Installer\vswhere.exe" -latest `
    -requires Microsoft.VisualStudio.Component.VC.Tools.ARM64 -property installationPath
Write-Host "Visual Studio: $vs"
$toolsets = Get-ChildItem "$vs\VC\Tools\MSVC" -Directory | Select-Object -ExpandProperty Name
Write-Host "MSVC toolsets: $($toolsets -join ', ')"

$src = (Resolve-Path "$PSScriptRoot\probe.cpp").Path
$out = (New-Item -ItemType Directory -Force probe-out).FullName
$devcmd = "$vs\Common7\Tools\VsDevCmd.bat"

foreach ($ts in $toolsets) {
    foreach ($opt in @('/Ox', '/O2')) {
        $tag = "msvc-$ts-$($opt.TrimStart('/'))"
        Write-Host "::group::compile $tag"
        cmd /c "call `"$devcmd`" -arch=arm64 -host_arch=arm64 -vcvars_ver=$ts -no_logo && cd /d `"$out`" && cl /nologo /std:c++17 /GR- /EHsc /MT $opt /FAs /Fa$tag.asm /Fe$tag.exe `"$src`""
        $built = $LASTEXITCODE
        Write-Host "::endgroup::"
        if ($built -ne 0) {
            Write-Host "::warning::$tag failed to compile"
            continue
        }
        Write-Host "===== $tag"
        & "$out\$tag.exe"
    }
}
