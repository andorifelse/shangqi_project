$ErrorActionPreference = "Stop"
$ProjectRoot = Split-Path -Parent $PSScriptRoot
Set-Location -LiteralPath $ProjectRoot
& "$ProjectRoot\.venv\Scripts\python.exe" -X utf8 "$ProjectRoot\tools\run_demo.py" @args
