$ErrorActionPreference = 'Stop'

$frontend = Join-Path $PSScriptRoot 'frontend'
$venvPython = Join-Path $PSScriptRoot '.venv\Scripts\python.exe'

if ($env:VIRTUAL_ENV) {
    $python = Join-Path $env:VIRTUAL_ENV 'Scripts\python.exe'
    if (-not (Test-Path -LiteralPath $python -PathType Leaf)) {
        throw 'The active virtual environment has no Python executable. Activate the correct environment and try again.'
    }
} elseif (Test-Path -LiteralPath $venvPython -PathType Leaf) {
    $python = $venvPython
} else {
    $launcher = Get-Command 'py.exe' -ErrorAction SilentlyContinue
    if (-not $launcher) {
        $launcher = Get-Command 'python.exe' -ErrorAction SilentlyContinue
    }
    if (-not $launcher) {
        throw 'Python was not found. See the installation steps in README.md.'
    }
    $python = $launcher.Source
}

$npm = Get-Command 'npm.cmd' -ErrorAction SilentlyContinue
if (-not $npm) {
    throw 'npm was not found. Install Node.js before starting the dashboard.'
}
if (-not (Test-Path -LiteralPath (Join-Path $frontend 'node_modules\.bin\next.cmd') -PathType Leaf)) {
    throw 'Frontend dependencies are missing. Run npm ci in the frontend directory first.'
}
& $python -m uvicorn --version | Out-Null
if ($LASTEXITCODE -ne 0) {
    throw "Uvicorn is unavailable in $python. Install the web extra in that Python environment first."
}

$listeners = [System.Net.NetworkInformation.IPGlobalProperties]::GetIPGlobalProperties().GetActiveTcpListeners()
foreach ($port in 8000, 3000) {
    if ($listeners.Port -contains $port) {
        throw "Port $port is already in use. Stop the existing server or application before running this script."
    }
}

$originalModelsDir = $env:ADAPTIVE_AI_MODELS_DIR
$originalApiUrl = $env:NEXT_PUBLIC_API_URL
try {
    if (-not $originalModelsDir) {
        $env:ADAPTIVE_AI_MODELS_DIR = Join-Path $PSScriptRoot 'runs'
    }
    if (-not $originalApiUrl) {
        $env:NEXT_PUBLIC_API_URL = 'http://127.0.0.1:8000'
    }

    Start-Process -FilePath $python -WorkingDirectory $PSScriptRoot `
        -ArgumentList '-m uvicorn backend.app.main:app --reload --host 127.0.0.1 --port 8000'
    Start-Process -FilePath $env:ComSpec -WorkingDirectory $frontend `
        -ArgumentList '/d /c npm.cmd run dev -- --hostname 127.0.0.1 --port 3000'
} finally {
    $env:ADAPTIVE_AI_MODELS_DIR = $originalModelsDir
    $env:NEXT_PUBLIC_API_URL = $originalApiUrl
}

Write-Host 'Starting both dev servers in separate windows:'
Write-Host '  Dashboard: http://127.0.0.1:3000'
Write-Host '  API docs:  http://127.0.0.1:8000/docs'
Write-Host 'Press Ctrl+C in each server window to stop it.'
