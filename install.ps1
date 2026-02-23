Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

function Write-MoveX {
    param([string]$Message)
    Write-Host "[MoveX] $Message" -ForegroundColor Cyan
}

function Resolve-Python {
    if (Get-Command py -ErrorAction SilentlyContinue) {
        return "py"
    }
    if (Get-Command python -ErrorAction SilentlyContinue) {
        return "python"
    }
    throw "Python 3.10+ is required. Install Python, then rerun this command."
}

function Ensure-Pipx {
    param([string]$PythonCmd)

    if (Get-Command pipx -ErrorAction SilentlyContinue) {
        return
    }

    Write-MoveX "Installing pipx..."
    & $PythonCmd -m pip install --user --upgrade pipx
    & $PythonCmd -m pipx ensurepath

    $userBase = (& $PythonCmd -c "import site; print(site.USER_BASE)").Trim()
    if (-not $userBase) {
        return
    }
    $pipxBin = Join-Path $userBase "Scripts"
    if ((Test-Path $pipxBin) -and -not ($env:Path.Split(';') -contains $pipxBin)) {
        $env:Path = "$pipxBin;$env:Path"
    }

    if (-not (Get-Command pipx -ErrorAction SilentlyContinue)) {
        throw "pipx was installed but is not available in PATH yet. Open a new terminal and run this command again."
    }
}

function Install-Or-Upgrade-MoveX {
    $installed = ""
    try {
        $listOutput = & pipx list --short 2>$null
        if ($LASTEXITCODE -eq 0 -and $null -ne $listOutput) {
            if ($listOutput -is [System.Array]) {
                $installed = $listOutput -join "`n"
            } else {
                $installed = [string]$listOutput
            }
        }
    } catch {
        $installed = ""
    }

    if ($installed -match "(?m)^movex\s") {
        Write-MoveX "Upgrading MoveX..."
        & pipx upgrade movex
        return
    }

    Write-MoveX "Installing MoveX..."
    & pipx install "git+https://github.com/MonishGosar/movex.git"
}

Write-MoveX "Starting installer..."
$python = Resolve-Python
Ensure-Pipx -PythonCmd $python
Install-Or-Upgrade-MoveX

Write-MoveX "Install complete. Run: movex --help"
