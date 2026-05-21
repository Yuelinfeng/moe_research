[CmdletBinding()]
param(
    [Parameter(Mandatory = $true)]
    [string]$SshCommand,

    [Parameter(Mandatory = $true)]
    [string]$RepoUrl,

    [string]$Config,
    [string]$LocalRepoPath,
    [string]$RemoteBaseDir = "/root/autodl-tmp",
    [string]$ResultsRoot,
    [string]$Branch = "main",
    [string]$SetupCommand = "",
    [string]$RunLabel = "smoke",
    [string]$RunCommand = 'if [[ -x experiments/run_baseline.sh ]]; then bash experiments/run_baseline.sh --output-dir "$RUN_DIR"; elif [[ -f experiments/run_baseline.sh ]]; then bash experiments/run_baseline.sh --output-dir "$RUN_DIR"; else git status --short > "$RUN_DIR/repo_status.txt"; find . -maxdepth 3 -type f | sort > "$RUN_DIR/repo_files.txt"; echo "No experiment entrypoint configured; wrote repo inventory."; fi',
    [switch]$InitGit,
    [switch]$Push
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

$ScriptRoot = if (-not [string]::IsNullOrWhiteSpace($PSScriptRoot)) {
    $PSScriptRoot
} else {
    Split-Path -Parent $MyInvocation.MyCommand.Path
}
$ProjectRoot = [System.IO.Path]::GetFullPath((Join-Path $ScriptRoot ".."))

if ([string]::IsNullOrWhiteSpace($Config)) {
    $Config = Join-Path $ProjectRoot "configs\remote.local.json"
}

function Resolve-LocalPath {
    param([string]$Path)
    if ([System.IO.Path]::IsPathRooted($Path)) {
        return [System.IO.Path]::GetFullPath($Path)
    }
    return [System.IO.Path]::GetFullPath((Join-Path (Get-Location) $Path))
}

function Invoke-Native {
    param(
        [string]$FilePath,
        [string[]]$ArgumentList,
        [string]$WorkingDirectory = (Get-Location).Path
    )

    Push-Location $WorkingDirectory
    try {
        & $FilePath @ArgumentList
        if ($LASTEXITCODE -ne 0) {
            throw "Command failed with exit code ${LASTEXITCODE}: $FilePath $($ArgumentList -join ' ')"
        }
    }
    finally {
        Pop-Location
    }
}

function Get-RepoName {
    param([string]$Url)

    $clean = $Url.Trim()
    if ($clean -match "\((https?://[^)]+)\)") {
        $clean = $Matches[1]
    }
    $leaf = ($clean -split "/")[-1]
    return ($leaf -replace "\.git$", "")
}

function Parse-SshCommand {
    param([string]$Command)

    $normalized = $Command.Trim()
    if ($normalized -notmatch "^ssh(\s+|$)") {
        throw "SshCommand must start with ssh."
    }

    $port = "22"
    if ($normalized -match "(^|\s)-p\s+([0-9]+)(\s|$)") {
        $port = $Matches[2]
    }

    $withoutSsh = $normalized -replace "^ssh\s+", ""
    $withoutPort = $withoutSsh -replace "(^|\s)-p\s+[0-9]+(\s|$)", " "
    $target = ($withoutPort.Trim() -split "\s+")[-1]

    if ($target -match "^([^@]+)@(.+)$") {
        return [pscustomobject]@{
            User = $Matches[1]
            Host = $Matches[2]
            Port = $port
        }
    }

    throw "Could not parse user@host from SshCommand: $Command"
}

$ssh = Parse-SshCommand $SshCommand
$repoName = Get-RepoName $RepoUrl
if ([string]::IsNullOrWhiteSpace($repoName)) {
    throw "Could not infer repo name from RepoUrl: $RepoUrl"
}

if ([string]::IsNullOrWhiteSpace($LocalRepoPath)) {
    $LocalRepoPath = $ProjectRoot
}
$LocalRepoPath = Resolve-LocalPath $LocalRepoPath

if ([string]::IsNullOrWhiteSpace($ResultsRoot)) {
    $ResultsRoot = Join-Path $ProjectRoot "results"
}
$ResultsRoot = Resolve-LocalPath $ResultsRoot

$remoteRepoPath = "$RemoteBaseDir/$repoName"
$remoteRunsRoot = "$RemoteBaseDir/runs/$repoName"

$configPayload = [ordered]@{
    local = [ordered]@{
        repoPath = $LocalRepoPath
        resultsRoot = $ResultsRoot
    }
    remote = [ordered]@{
        host = $ssh.Host
        user = $ssh.User
        port = $ssh.Port
        repoUrl = $RepoUrl
        repoPath = $remoteRepoPath
        runsRoot = $remoteRunsRoot
        setupCommand = $SetupCommand
    }
    git = [ordered]@{
        remote = "origin"
        branch = $Branch
    }
    experiment = [ordered]@{
        runLabel = $RunLabel
        runCommand = $RunCommand
    }
}

$configPath = Resolve-LocalPath $Config
New-Item -ItemType Directory -Force -Path (Split-Path -Parent $configPath) | Out-Null
$configPayload | ConvertTo-Json -Depth 6 | Set-Content -LiteralPath $configPath -Encoding UTF8
Write-Host "Wrote config: $configPath"

if ($InitGit) {
    if (-not (Test-Path -LiteralPath (Join-Path $LocalRepoPath ".git"))) {
        Invoke-Native -FilePath git -ArgumentList @("init", "-b", $Branch) -WorkingDirectory $LocalRepoPath
    } else {
        $currentBranch = (& git -C $LocalRepoPath branch --show-current)
        if ($currentBranch.Trim() -ne $Branch) {
            Invoke-Native -FilePath git -ArgumentList @("checkout", "-B", $Branch) -WorkingDirectory $LocalRepoPath
        }
    }

    $remoteNames = @(& git -C $LocalRepoPath remote)
    if ($remoteNames -notcontains "origin") {
        Invoke-Native -FilePath git -ArgumentList @("remote", "add", "origin", $RepoUrl) -WorkingDirectory $LocalRepoPath
    } else {
        $remoteUrl = (& git -C $LocalRepoPath remote get-url origin)
        if ($remoteUrl.Trim() -ne $RepoUrl) {
        Invoke-Native -FilePath git -ArgumentList @("remote", "set-url", "origin", $RepoUrl) -WorkingDirectory $LocalRepoPath
        }
    }
}

if ($Push) {
    Invoke-Native -FilePath git -ArgumentList @("add", "-A") -WorkingDirectory $LocalRepoPath
    $status = (& git -C $LocalRepoPath status --short)
    if ($status) {
        Invoke-Native -FilePath git -ArgumentList @("commit", "-m", "initialize remote experiment scaffold") -WorkingDirectory $LocalRepoPath
    }
    Invoke-Native -FilePath git -ArgumentList @("push", "-u", "origin", $Branch) -WorkingDirectory $LocalRepoPath
}

Write-Host "Next run command:"
Write-Host "powershell -ExecutionPolicy Bypass -File $ProjectRoot\scripts\local_push_run_fetch.ps1 -Config $configPath"
