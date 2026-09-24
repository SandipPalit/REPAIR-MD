param(
    [string]$Engine = 'tools\diffuse-cpp\build\diffuse-cli.exe'
)

$missing = @()
foreach ($command in @('cmake', 'git')) {
    if (-not (Get-Command $command -ErrorAction SilentlyContinue)) {
        $missing += $command
    }
}
$compiler = Get-Command cl, clang++, g++ -ErrorAction SilentlyContinue | Select-Object -First 1
if (-not $compiler) {
    $missing += 'C++17 compiler (MSVC cl.exe, clang++, or g++)'
}

if ($missing.Count -gt 0) {
    Write-Error ('Cannot build diffuse-cpp. Missing: ' + ($missing -join ', ') + '. Install CMake and the Visual Studio C++ build tools, then run this script from a Developer PowerShell.')
    exit 2
}

if (-not (Test-Path 'tools\diffuse-cpp')) {
    git clone --depth 1 https://github.com/iafiscal1212/diffuse-cpp.git tools\diffuse-cpp
}
cmake -S tools\diffuse-cpp -B tools\diffuse-cpp\build -DCMAKE_BUILD_TYPE=Release
cmake --build tools\diffuse-cpp\build --config Release --parallel

$built = Get-ChildItem tools\diffuse-cpp\build -Recurse -Filter diffuse-cli.exe | Select-Object -First 1
if (-not $built) {
    Write-Error 'Build completed without producing diffuse-cli.exe.'
    exit 3
}
Write-Host "diffuse-cli ready: $($built.FullName)"