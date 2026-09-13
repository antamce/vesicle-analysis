[CmdletBinding()]
param(
    [switch]$SelfTest,
    [Parameter(ValueFromRemainingArguments = $true)]
    [string[]]$AppArguments
)

$ErrorActionPreference = "Stop"
$ProjectRoot = Split-Path -Parent $PSScriptRoot
$EnvironmentFile = Join-Path $ProjectRoot "environment.yml"
$WindowsIcon = Join-Path $PSScriptRoot "assets\icon.ico"
$BatchLauncher = Join-Path $ProjectRoot "Launch Vesicle Analysis.bat"
$ConfigDirectory = Join-Path $env:LOCALAPPDATA "VesicleAnalysis"
$ConfigFile = Join-Path $ConfigDirectory "launcher.json"
$MarkerName = "vesicle-analysis-launcher.marker"
$MarkerValue = "vesicle-analysis-project-environment-v1"

function Write-Step([string]$Message) {
    Write-Host "[Vesicle Analysis] $Message" -ForegroundColor Cyan
}

function Read-LauncherConfig {
    if (-not (Test-Path -LiteralPath $ConfigFile -PathType Leaf)) {
        return [pscustomobject]@{}
    }
    try {
        return Get-Content -LiteralPath $ConfigFile -Raw | ConvertFrom-Json
    }
    catch {
        Write-Host "The saved launcher configuration was invalid and will be replaced." -ForegroundColor Yellow
        return [pscustomobject]@{}
    }
}

function Save-LauncherConfig(
    [string]$CondaPath,
    [string]$EnvironmentName,
    [string]$EnvironmentHash
) {
    New-Item -ItemType Directory -Path $ConfigDirectory -Force | Out-Null
    [ordered]@{
        CondaPath = $CondaPath
        EnvironmentName = $EnvironmentName
        EnvironmentHash = $EnvironmentHash
    } | ConvertTo-Json | Set-Content -LiteralPath $ConfigFile -Encoding UTF8
}

function Test-CondaExecutable([string]$Candidate) {
    if (-not $Candidate -or -not (Test-Path -LiteralPath $Candidate -PathType Leaf)) {
        return $false
    }
    try {
        $version = & $Candidate --version 2>&1
        return $LASTEXITCODE -eq 0 -and "$version" -match "conda"
    }
    catch {
        return $false
    }
}

function Resolve-CondaCandidate([string]$Candidate) {
    if ([string]::IsNullOrWhiteSpace($Candidate)) {
        return $null
    }
    $candidatePath = [Environment]::ExpandEnvironmentVariables(
        $Candidate.Trim().Trim('"').Trim("'")
    )
    $possibilities = [System.Collections.Generic.List[string]]::new()
    if (Test-Path -LiteralPath $candidatePath -PathType Container) {
        foreach ($relative in @(
            "Scripts\conda.exe",
            "condabin\conda.bat",
            "bin\conda.exe",
            "bin\conda"
        )) {
            $possibilities.Add((Join-Path $candidatePath $relative))
        }
    }
    elseif (Test-Path -LiteralPath $candidatePath -PathType Leaf) {
        $leaf = Split-Path -Leaf $candidatePath
        $parent = Split-Path -Parent $candidatePath
        $root = Split-Path -Parent $parent
        if ($leaf -ieq "conda.bat") {
            $possibilities.Add((Join-Path $root "Scripts\conda.exe"))
        }
        $possibilities.Add($candidatePath)
    }
    foreach ($path in $possibilities | Select-Object -Unique) {
        if (Test-CondaExecutable $path) {
            return (Resolve-Path -LiteralPath $path).Path
        }
    }
    return $null
}

function Find-CondaAutomatically($Config) {
    $candidates = [System.Collections.Generic.List[string]]::new()
    if ($Config.PSObject.Properties.Name -contains "CondaPath") {
        $candidates.Add("$($Config.CondaPath)")
    }
    if ($env:CONDA_EXE) {
        $candidates.Add($env:CONDA_EXE)
    }
    foreach ($commandName in @("conda.exe", "conda.bat", "conda")) {
        $command = Get-Command $commandName -ErrorAction SilentlyContinue | Select-Object -First 1
        if ($command -and $command.Source) {
            $candidates.Add($command.Source)
        }
    }
    foreach ($root in @(
        (Join-Path $env:USERPROFILE "anaconda3"),
        (Join-Path $env:USERPROFILE "miniconda3"),
        (Join-Path $env:USERPROFILE "miniforge3"),
        (Join-Path $env:USERPROFILE "mambaforge"),
        (Join-Path $env:LOCALAPPDATA "anaconda3"),
        (Join-Path $env:LOCALAPPDATA "miniconda3"),
        (Join-Path $env:LOCALAPPDATA "miniforge3"),
        (Join-Path $env:ProgramData "anaconda3"),
        (Join-Path $env:ProgramData "miniconda3"),
        (Join-Path $env:ProgramData "miniforge3"),
        "C:\Program Files\Anaconda3",
        "C:\Program Files\Miniconda3",
        "C:\Program Files\Miniforge3"
    )) {
        $candidates.Add($root)
    }
    foreach ($registryRoot in @(
        "HKCU:\Software\Microsoft\Windows\CurrentVersion\Uninstall\*",
        "HKLM:\Software\Microsoft\Windows\CurrentVersion\Uninstall\*",
        "HKLM:\Software\WOW6432Node\Microsoft\Windows\CurrentVersion\Uninstall\*"
    )) {
        Get-ItemProperty $registryRoot -ErrorAction SilentlyContinue |
            Where-Object { $_.DisplayName -match "Anaconda|Miniconda|Miniforge|Mambaforge" } |
            ForEach-Object {
                if ($_.InstallLocation) {
                    $candidates.Add($_.InstallLocation)
                }
            }
    }
    foreach ($candidate in $candidates | Select-Object -Unique) {
        $resolved = Resolve-CondaCandidate $candidate
        if ($resolved) {
            return $resolved
        }
    }
    return $null
}

function Request-CondaPath {
    Add-Type -AssemblyName System.Windows.Forms
    while ($true) {
        $choice = [System.Windows.Forms.MessageBox]::Show(
            "Conda was not found automatically.`n`nChoose Yes to select conda.exe/conda.bat, No to select an Anaconda, Miniconda, Miniforge, or Mambaforge installation folder, or Cancel to stop.`n`nIf Conda is not installed, install one of those distributions and run this launcher again.",
            "Locate Conda",
            [System.Windows.Forms.MessageBoxButtons]::YesNoCancel,
            [System.Windows.Forms.MessageBoxIcon]::Information
        )
        if ($choice -eq [System.Windows.Forms.DialogResult]::Cancel) {
            return $null
        }
        $selected = $null
        if ($choice -eq [System.Windows.Forms.DialogResult]::Yes) {
            $dialog = [System.Windows.Forms.OpenFileDialog]::new()
            $dialog.Title = "Select the Conda executable"
            $dialog.Filter = "Conda (conda.exe;conda.bat)|conda.exe;conda.bat|All files|*.*"
            if ($dialog.ShowDialog() -eq [System.Windows.Forms.DialogResult]::OK) {
                $selected = $dialog.FileName
            }
        }
        else {
            $dialog = [System.Windows.Forms.FolderBrowserDialog]::new()
            $dialog.Description = "Select the Conda installation folder"
            if ($dialog.ShowDialog() -eq [System.Windows.Forms.DialogResult]::OK) {
                $selected = $dialog.SelectedPath
            }
        }
        if (-not $selected) {
            continue
        }
        $resolved = Resolve-CondaCandidate $selected
        if ($resolved) {
            return $resolved
        }
        [System.Windows.Forms.MessageBox]::Show(
            "That path does not contain a working Conda executable. Select conda.exe/conda.bat or the top-level installation folder and try again.",
            "Invalid Conda path",
            [System.Windows.Forms.MessageBoxButtons]::OK,
            [System.Windows.Forms.MessageBoxIcon]::Warning
        ) | Out-Null
    }
}

function Get-EnvironmentPrefix([string]$CondaPath, [string]$Name) {
    if ([string]::IsNullOrWhiteSpace($Name)) {
        return $null
    }
    try {
        # Do not parse the JSON form with Windows PowerShell here. Conda can
        # include paths that differ only by letter case as keys in
        # envs_details. ConvertFrom-Json treats those keys as duplicates and
        # rejects the entire response. The stable text table is sufficient
        # and also detects an existing environment even if its Python cannot
        # be started, preserving the launcher's no-overwrite safety rule.
        $lines = @(& $CondaPath env list 2>$null)
        $exitCode = $LASTEXITCODE
        if ($exitCode -ne 0) {
            return $null
        }
        $environmentLine = '^\s*(?<name>[^\s#]+)\s+(?:(?:\*|\+)\s+)?(?<prefix>(?:[A-Za-z]:[\\/]|\\\\|/).+?)\s*$'
        foreach ($line in $lines) {
            if ("$line" -match $environmentLine -and $Matches.name -ieq $Name) {
                return $Matches.prefix
            }
        }
    }
    catch {
        return $null
    }
    return $null
}

function Test-ManagedEnvironment([string]$Prefix) {
    if (-not $Prefix) {
        return $false
    }
    $marker = Join-Path $Prefix "conda-meta\$MarkerName"
    return (Test-Path -LiteralPath $marker -PathType Leaf) -and
        ((Get-Content -LiteralPath $marker -Raw).Trim() -eq $MarkerValue)
}

function Select-EnvironmentName([string]$CondaPath, $Config) {
    if ($Config.PSObject.Properties.Name -contains "EnvironmentName") {
        $savedName = "$($Config.EnvironmentName)"
        $savedPrefix = Get-EnvironmentPrefix $CondaPath $savedName
        if (Test-ManagedEnvironment $savedPrefix) {
            return $savedName
        }
    }
    $names = @("vesicle-analysis", "vesicle-analysis-app")
    for ($index = 2; $index -le 99; $index++) {
        $names += "vesicle-analysis-app-$index"
    }
    foreach ($name in $names) {
        $prefix = Get-EnvironmentPrefix $CondaPath $name
        if (-not $prefix -or (Test-ManagedEnvironment $prefix)) {
            return $name
        }
    }
    throw "Could not choose an unused managed environment name."
}

function Set-EnvironmentMarker([string]$Prefix) {
    if (-not $Prefix) {
        throw "Conda did not report the new environment path."
    }
    $marker = Join-Path $Prefix "conda-meta\$MarkerName"
    Set-Content -LiteralPath $marker -Value $MarkerValue -Encoding ASCII
}

function Ensure-ManagedEnvironment(
    [string]$CondaPath,
    [string]$EnvironmentName,
    [string]$SavedHash,
    [string]$CurrentHash
) {
    $prefix = Get-EnvironmentPrefix $CondaPath $EnvironmentName
    if (-not $prefix) {
        Write-Step "Creating the dedicated '$EnvironmentName' environment. Internet access may be required."
        & $CondaPath env create --name $EnvironmentName --file $EnvironmentFile --yes | Out-Host
        if ($LASTEXITCODE -ne 0) {
            throw "Conda could not create the environment. Check the internet connection and messages above, then try again."
        }
        $prefix = Get-EnvironmentPrefix $CondaPath $EnvironmentName
        if (-not $prefix) {
            throw "Conda reported successful creation, but '$EnvironmentName' could not be opened. Run 'conda env list' in an Anaconda Prompt and check the messages above."
        }
        Set-EnvironmentMarker $prefix
        return $prefix
    }
    if (-not (Test-ManagedEnvironment $prefix)) {
        throw "Safety check failed: '$EnvironmentName' exists but is not owned by this launcher."
    }
    if ($SavedHash -ne $CurrentHash) {
        Write-Step "Updating '$EnvironmentName' because environment.yml changed. Internet access may be required."
        & $CondaPath env update --name $EnvironmentName --file $EnvironmentFile --prune --yes | Out-Host
        if ($LASTEXITCODE -ne 0) {
            throw "Conda could not update the environment. Check the internet connection and messages above, then try again."
        }
    }
    return $prefix
}

function New-DesktopShortcut {
    $desktop = [Environment]::GetFolderPath("DesktopDirectory")
    if (-not $desktop) {
        throw "Windows did not report a Desktop folder."
    }
    $shortcutPath = Join-Path $desktop "Vesicle Analysis.lnk"
    $shell = New-Object -ComObject WScript.Shell
    $shortcut = $shell.CreateShortcut($shortcutPath)
    $shortcut.TargetPath = $BatchLauncher
    $shortcut.WorkingDirectory = $ProjectRoot
    $shortcut.Description = "Launch Vesicle Analysis"
    if (Test-Path -LiteralPath $WindowsIcon -PathType Leaf) {
        $shortcut.IconLocation = "$WindowsIcon,0"
    }
    $shortcut.Save()
    Write-Step "Desktop shortcut ready: $shortcutPath"
}

function Quote-NativeArgument([string]$Value) {
    if ($Value -notmatch '[\s"]') {
        return $Value
    }
    return '"' + $Value.Replace('"', '\"') + '"'
}

function Start-VesicleAnalysis(
    [string]$CondaPath,
    [string]$EnvironmentName,
    [string[]]$Arguments
) {
    New-Item -ItemType Directory -Path $ConfigDirectory -Force | Out-Null
    $standardLog = Join-Path $ConfigDirectory "app-output.log"
    $errorLog = Join-Path $ConfigDirectory "app-error.log"
    $nativeArguments = @(
        "run",
        "--no-capture-output",
        "--name",
        $EnvironmentName,
        "python",
        "-m",
        "vesicle_analysis"
    ) + $Arguments
    $argumentLine = ($nativeArguments | ForEach-Object { Quote-NativeArgument "$_" }) -join " "
    return Start-Process -FilePath $CondaPath `
        -ArgumentList $argumentLine `
        -WorkingDirectory $ProjectRoot `
        -WindowStyle Hidden `
        -RedirectStandardOutput $standardLog `
        -RedirectStandardError $errorLog `
        -PassThru
}

try {
    if (-not (Test-Path -LiteralPath $EnvironmentFile -PathType Leaf)) {
        throw "environment.yml is missing from $ProjectRoot."
    }
    $config = Read-LauncherConfig
    $condaPath = Find-CondaAutomatically $config
    if (-not $condaPath -and -not $SelfTest) {
        $condaPath = Request-CondaPath
    }
    if (-not $condaPath) {
        throw "Conda was not selected. Install Anaconda, Miniconda, Miniforge, or Mambaforge, then run the launcher again."
    }
    Write-Step "Using Conda: $condaPath"
    if ($SelfTest) {
        Write-Step "Self-test passed. Project files and Conda detection are working."
        exit 0
    }

    $previousEnvironmentName = ""
    $previousEnvironmentHash = ""
    if ($config.PSObject.Properties.Name -contains "EnvironmentName") {
        $previousEnvironmentName = "$($config.EnvironmentName)"
    }
    if ($config.PSObject.Properties.Name -contains "EnvironmentHash") {
        $previousEnvironmentHash = "$($config.EnvironmentHash)"
    }
    Save-LauncherConfig $condaPath $previousEnvironmentName $previousEnvironmentHash

    $environmentName = Select-EnvironmentName $condaPath $config
    $savedHash = ""
    if ($config.PSObject.Properties.Name -contains "EnvironmentHash" -and
        $config.PSObject.Properties.Name -contains "EnvironmentName" -and
        "$($config.EnvironmentName)" -eq $environmentName) {
        $savedHash = "$($config.EnvironmentHash)"
    }
    $environmentHash = (Get-FileHash -LiteralPath $EnvironmentFile -Algorithm SHA256).Hash.ToLowerInvariant()
    $prefix = Ensure-ManagedEnvironment $condaPath $environmentName $savedHash $environmentHash
    if (-not (Test-ManagedEnvironment $prefix)) {
        throw "The managed-environment ownership marker could not be verified."
    }
    Save-LauncherConfig $condaPath $environmentName $environmentHash
    New-DesktopShortcut
    Write-Step "Starting Vesicle Analysis. Setup output will close; application errors are logged in $ConfigDirectory."
    $appProcess = Start-VesicleAnalysis $condaPath $environmentName $AppArguments
    Start-Sleep -Seconds 2
    if ($appProcess.HasExited -and $appProcess.ExitCode -ne 0) {
        $details = ""
        $errorLog = Join-Path $ConfigDirectory "app-error.log"
        if (Test-Path -LiteralPath $errorLog -PathType Leaf) {
            $details = (Get-Content -LiteralPath $errorLog -Tail 20) -join [Environment]::NewLine
        }
        throw "The application exited during startup.`n$details"
    }
    exit 0
}
catch {
    Write-Host ""
    Write-Host "Vesicle Analysis launcher error:" -ForegroundColor Red
    Write-Host $_.Exception.Message -ForegroundColor Red
    Write-Host ""
    Write-Host "If Conda is not installed, install Anaconda, Miniconda, Miniforge, or Mambaforge and try again."
    exit 1
}
