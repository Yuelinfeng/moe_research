[CmdletBinding()]
param(
    [string]$Config,
    [string]$RepoLocalPath,
    [string]$RemoteHost,
    [string]$RemoteUser,
    [string]$RemotePort,
    [string]$RemoteRepoUrl,
    [string]$RemoteRepoPath,
    [string]$RemoteRunsRoot,
    [string]$ResultsRoot,
    [string]$GitRemote,
    [string]$Branch,
    [string]$RunLabel,
    [string]$RunCommand,
    [string]$SetupCommand,
    [string]$CommitMessage,
    [switch]$NoPush,
    [switch]$NoFetch,
    [switch]$Help
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

$ScriptRoot = if (-not [string]::IsNullOrWhiteSpace($PSScriptRoot)) {
    $PSScriptRoot
} else {
    Split-Path -Parent $MyInvocation.MyCommand.Path
}

if ([string]::IsNullOrWhiteSpace($Config)) {
    $Config = Join-Path $ScriptRoot "..\configs\remote.example.json"
}

function Show-Help {
    $readme = Join-Path $ScriptRoot "..\README.md"
    Get-Content -LiteralPath $readme
}

function Require-Command {
    param([string]$Name)
    if (-not (Get-Command $Name -ErrorAction SilentlyContinue)) {
        throw "Required command not found: $Name"
    }
}

function Invoke-Native {
    param(
        [string]$FilePath,
        [string[]]$ArgumentList
    )
    & $FilePath @ArgumentList
    if ($LASTEXITCODE -ne 0) {
        throw "Command failed with exit code ${LASTEXITCODE}: $FilePath $($ArgumentList -join ' ')"
    }
}

function Get-NestedValue {
    param(
        [AllowNull()][object]$Object,
        [string]$Path
    )

    $current = $Object
    foreach ($part in $Path.Split(".")) {
        if ($null -eq $current) {
            return $null
        }
        $property = $current.PSObject.Properties[$part]
        if ($null -eq $property) {
            return $null
        }
        $current = $property.Value
    }
    return $current
}

function Get-ConfigString {
    param(
        [AllowNull()][string]$CliValue,
        [AllowNull()][object]$ConfigObject,
        [string]$Path,
        [AllowNull()][string]$Default
    )

    if (-not [string]::IsNullOrWhiteSpace($CliValue)) {
        return $CliValue
    }

    $value = Get-NestedValue -Object $ConfigObject -Path $Path
    if ($null -ne $value -and -not [string]::IsNullOrWhiteSpace([string]$value)) {
        return [string]$value
    }

    return $Default
}

function Resolve-LocalPath {
    param([string]$Path)
    if ([System.IO.Path]::IsPathRooted($Path)) {
        return [System.IO.Path]::GetFullPath($Path)
    }
    return [System.IO.Path]::GetFullPath((Join-Path (Get-Location) $Path))
}

function ConvertTo-ShellLiteral {
    param([AllowNull()][string]$Value)
    if ($null -eq $Value) {
        return "''"
    }
    return "'" + ($Value -replace "'", "'\''") + "'"
}

if ($Help) {
    Show-Help
    exit 0
}

Require-Command git
Require-Command ssh
Require-Command scp

$configPath = Resolve-Path -LiteralPath $Config
$configObject = Get-Content -Raw -LiteralPath $configPath | ConvertFrom-Json

$RepoLocalPath = Resolve-LocalPath (Get-ConfigString $RepoLocalPath $configObject "local.repoPath" (Get-Location).Path)
$ResultsRoot = Resolve-LocalPath (Get-ConfigString $ResultsRoot $configObject "local.resultsRoot" (Join-Path (Get-Location) "results"))
$RemoteHost = Get-ConfigString $RemoteHost $configObject "remote.host" $null
$RemoteUser = Get-ConfigString $RemoteUser $configObject "remote.user" ""
$RemotePort = Get-ConfigString $RemotePort $configObject "remote.port" ""
$RemoteRepoUrl = Get-ConfigString $RemoteRepoUrl $configObject "remote.repoUrl" ""
$RemoteRepoPath = Get-ConfigString $RemoteRepoPath $configObject "remote.repoPath" $null
$RemoteRunsRoot = Get-ConfigString $RemoteRunsRoot $configObject "remote.runsRoot" "/root/autodl-tmp/runs"
$SetupCommand = Get-ConfigString $SetupCommand $configObject "remote.setupCommand" ""
$GitRemote = Get-ConfigString $GitRemote $configObject "git.remote" "origin"
$Branch = Get-ConfigString $Branch $configObject "git.branch" $null
$RunLabel = Get-ConfigString $RunLabel $configObject "experiment.runLabel" "moe_run"
$RunCommand = Get-ConfigString $RunCommand $configObject "experiment.runCommand" $null

if ([string]::IsNullOrWhiteSpace($RemoteHost)) {
    throw "Missing remote.host. Set it in config or pass -RemoteHost."
}
if ([string]::IsNullOrWhiteSpace($RemoteRepoPath)) {
    throw "Missing remote.repoPath. Set it in config or pass -RemoteRepoPath."
}
if ([string]::IsNullOrWhiteSpace($RunCommand)) {
    throw "Missing experiment.runCommand. Set it in config or pass -RunCommand."
}

$RemoteTarget = if ([string]::IsNullOrWhiteSpace($RemoteUser)) {
    $RemoteHost
} else {
    "${RemoteUser}@${RemoteHost}"
}
$SshPrefixArgs = @()
$ScpPrefixArgs = @("-O")
if (-not [string]::IsNullOrWhiteSpace($RemotePort)) {
    $SshPrefixArgs += @("-p", $RemotePort)
    $ScpPrefixArgs += @("-P", $RemotePort)
}

Push-Location $RepoLocalPath
try {
    $inside = & git rev-parse --is-inside-work-tree 2>$null
    if ($LASTEXITCODE -ne 0 -or $inside.Trim() -ne "true") {
        throw "Local repo path is not a git repository: $RepoLocalPath"
    }

    if ([string]::IsNullOrWhiteSpace($Branch)) {
        $Branch = (& git branch --show-current).Trim()
        if ($LASTEXITCODE -ne 0 -or [string]::IsNullOrWhiteSpace($Branch)) {
            throw "Could not infer current git branch. Pass -Branch explicitly."
        }
    }

    $status = & git status --short
    if ($LASTEXITCODE -ne 0) {
        throw "Could not read git status in $RepoLocalPath"
    }

    if ($status) {
        if ([string]::IsNullOrWhiteSpace($CommitMessage)) {
            throw "Local repo has uncommitted changes. Commit them first or pass -CommitMessage."
        }
        Invoke-Native git @("add", "-A")
        Invoke-Native git @("commit", "-m", $CommitMessage)
    }

    $localCommit = (& git rev-parse HEAD).Trim()
    if ($LASTEXITCODE -ne 0) {
        throw "Could not read local commit hash."
    }

    if (-not $NoPush) {
        Invoke-Native git @("push", $GitRemote, $Branch)
    }
}
finally {
    Pop-Location
}

$remoteScriptLocal = Join-Path $ScriptRoot "remote_pull_run.sh"
if (-not (Test-Path -LiteralPath $remoteScriptLocal)) {
    throw "Missing remote runner: $remoteScriptLocal"
}

$guid = [guid]::NewGuid().ToString("N")
$localRemoteConfig = Join-Path ([System.IO.Path]::GetTempPath()) "airs_remote_config_$guid.env"
$remoteScriptPath = "/tmp/airs_remote_pull_run_$guid.sh"
$remoteConfigPath = "/tmp/airs_remote_config_$guid.env"

$remoteRunConfig = [ordered]@{
    REPO_PATH = $RemoteRepoPath
    RUNS_ROOT = $RemoteRunsRoot
    GIT_REMOTE = $GitRemote
    BRANCH = $Branch
    RUN_LABEL = $RunLabel
    RUN_COMMAND = $RunCommand
    SETUP_COMMAND = $SetupCommand
    EXPECTED_COMMIT = $localCommit
    REPO_URL = $RemoteRepoUrl
}

$localRunPath = $null
try {
    $configLines = $remoteRunConfig.GetEnumerator() | ForEach-Object {
        "export $($_.Key)=$(ConvertTo-ShellLiteral ([string]$_.Value))"
    }
    $configLines | Set-Content -LiteralPath $localRemoteConfig -Encoding ASCII

    Invoke-Native scp @($ScpPrefixArgs + @($remoteScriptLocal, "${RemoteTarget}:$remoteScriptPath"))
    Invoke-Native scp @($ScpPrefixArgs + @($localRemoteConfig, "${RemoteTarget}:$remoteConfigPath"))

    $remoteCommand = "chmod +x $remoteScriptPath && bash $remoteScriptPath $remoteConfigPath; status=`$?; rm -f $remoteScriptPath $remoteConfigPath; exit `$status"
    $outputLines = New-Object "System.Collections.Generic.List[string]"

    $previousErrorActionPreference = $ErrorActionPreference
    $ErrorActionPreference = "Continue"
    try {
        & ssh @SshPrefixArgs $RemoteTarget $remoteCommand 2>&1 | ForEach-Object {
            $line = $_.ToString()
            $outputLines.Add($line)
            Write-Host $line
        }
        $remoteStatus = $LASTEXITCODE
    }
    finally {
        $ErrorActionPreference = $previousErrorActionPreference
    }

    $runIdLine = $outputLines | Where-Object { $_ -match "^AIRS_RUN_ID=" } | Select-Object -Last 1
    $runDirLine = $outputLines | Where-Object { $_ -match "^AIRS_RUN_DIR=" } | Select-Object -Last 1
    $runId = if ($runIdLine) { $runIdLine -replace "^AIRS_RUN_ID=", "" } else { $null }
    $runDir = if ($runDirLine) { $runDirLine -replace "^AIRS_RUN_DIR=", "" } else { $null }

    if (-not $NoFetch) {
        if ([string]::IsNullOrWhiteSpace($runDir) -or [string]::IsNullOrWhiteSpace($runId)) {
            throw "Remote run did not report AIRS_RUN_ID/AIRS_RUN_DIR; cannot fetch results."
        }
        New-Item -ItemType Directory -Force -Path $ResultsRoot | Out-Null
        Invoke-Native scp @($ScpPrefixArgs + @("-r", "${RemoteTarget}:$runDir", $ResultsRoot))
        $localRunPath = Join-Path $ResultsRoot $runId
        Write-Host "AIRS_LOCAL_RESULT=$localRunPath"
    }

    if ($remoteStatus -ne 0) {
        if ($localRunPath) {
            throw "Remote run failed with exit code $remoteStatus. Output was fetched to $localRunPath"
        }
        throw "Remote run failed with exit code $remoteStatus."
    }
}
finally {
    if (Test-Path -LiteralPath $localRemoteConfig) {
        Remove-Item -LiteralPath $localRemoteConfig -Force
    }
}
