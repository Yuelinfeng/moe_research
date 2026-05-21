[CmdletBinding()]
param(
    [string]$TargetPath = "D:\moe_research",
    [string]$RepoUrl = "https://github.com/Yuelinfeng/moe_research.git",
    [string]$Branch = "main",
    [switch]$Overwrite,
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
$SourceRoot = [System.IO.Path]::GetFullPath((Join-Path $ScriptRoot ".."))
$TargetRoot = [System.IO.Path]::GetFullPath($TargetPath)

if ($SourceRoot.TrimEnd("\") -ieq $TargetRoot.TrimEnd("\")) {
    throw "Source and target are the same directory: $TargetRoot"
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

New-Item -ItemType Directory -Force -Path $TargetRoot | Out-Null

$existing = Get-ChildItem -Force -LiteralPath $TargetRoot | Where-Object {
    $_.Name -ne ".git"
}
if ($existing -and -not $Overwrite) {
    throw "Target is not empty: $TargetRoot. Re-run with -Overwrite to merge files into it."
}

$excludeNames = @(".git")
Get-ChildItem -Force -LiteralPath $SourceRoot | Where-Object {
    $excludeNames -notcontains $_.Name
} | ForEach-Object {
    $destination = Join-Path $TargetRoot $_.Name
    Copy-Item -LiteralPath $_.FullName -Destination $destination -Recurse -Force
}

$localConfig = Join-Path $TargetRoot "configs\remote.local.json"
if (Test-Path -LiteralPath $localConfig) {
    $cfg = Get-Content -Raw -LiteralPath $localConfig | ConvertFrom-Json
    $cfg.local.repoPath = $TargetRoot
    $cfg.local.resultsRoot = Join-Path $TargetRoot "results"
    $cfg | ConvertTo-Json -Depth 8 | Set-Content -LiteralPath $localConfig -Encoding UTF8
}

if ($InitGit) {
    if (-not (Test-Path -LiteralPath (Join-Path $TargetRoot ".git"))) {
        Invoke-Native -FilePath git -ArgumentList @("init", "-b", $Branch) -WorkingDirectory $TargetRoot
    }

    $remotes = @(& git -C $TargetRoot remote)
    if ($remotes -notcontains "origin") {
        Invoke-Native -FilePath git -ArgumentList @("remote", "add", "origin", $RepoUrl) -WorkingDirectory $TargetRoot
    } else {
        $currentRemote = (& git -C $TargetRoot remote get-url origin).Trim()
        if ($currentRemote -ne $RepoUrl) {
            Invoke-Native -FilePath git -ArgumentList @("remote", "set-url", "origin", $RepoUrl) -WorkingDirectory $TargetRoot
        }
    }
}

if ($Push) {
    if (-not (Test-Path -LiteralPath (Join-Path $TargetRoot ".git"))) {
        throw "Cannot push because target is not a git repository. Add -InitGit."
    }

    Invoke-Native -FilePath git -ArgumentList @("add", "-A") -WorkingDirectory $TargetRoot
    $status = (& git -C $TargetRoot status --short)
    if ($status) {
        Invoke-Native -FilePath git -ArgumentList @("commit", "-m", "initialize moe research scaffold") -WorkingDirectory $TargetRoot
    }
    Invoke-Native -FilePath git -ArgumentList @("push", "-u", "origin", $Branch) -WorkingDirectory $TargetRoot
}

Write-Host "Migrated project to $TargetRoot"
Write-Host "Next command:"
Write-Host "cd $TargetRoot"
