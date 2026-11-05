$ErrorActionPreference = 'Stop'
$projectRoot = Split-Path -Parent $PSScriptRoot
$pythonExe = Join-Path $projectRoot '.venv\Scripts\python.exe'
if (-not (Test-Path $pythonExe)) { $pythonExe = 'python' }
& $pythonExe (Join-Path $PSScriptRoot 'build.py') --platform windows @args
exit $LASTEXITCODE
