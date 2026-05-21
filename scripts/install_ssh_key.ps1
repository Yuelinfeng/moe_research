[CmdletBinding()]
param(
    [Parameter(Mandatory = $true)]
    [string]$SshCommand,

    [string]$KeyPath = "$env:USERPROFILE\.ssh\id_ed25519"
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

function Require-Command {
    param([string]$Name)
    if (-not (Get-Command $Name -ErrorAction SilentlyContinue)) {
        throw "Required command not found: $Name"
    }
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

Require-Command ssh
Require-Command ssh-keygen

$ssh = Parse-SshCommand $SshCommand
$keyDir = Split-Path -Parent $KeyPath
New-Item -ItemType Directory -Force -Path $keyDir | Out-Null

if (-not (Test-Path -LiteralPath $KeyPath)) {
    & ssh-keygen -t ed25519 -N "" -f $KeyPath
    if ($LASTEXITCODE -ne 0) {
        throw "ssh-keygen failed with exit code $LASTEXITCODE"
    }
}

$publicKeyPath = "$KeyPath.pub"
if (-not (Test-Path -LiteralPath $publicKeyPath)) {
    throw "Missing public key: $publicKeyPath"
}

$publicKey = Get-Content -Raw -LiteralPath $publicKeyPath
$target = "$($ssh.User)@$($ssh.Host)"
$remoteCommand = "mkdir -p ~/.ssh && chmod 700 ~/.ssh && touch ~/.ssh/authorized_keys && grep -qxF '$publicKey' ~/.ssh/authorized_keys || echo '$publicKey' >> ~/.ssh/authorized_keys && chmod 600 ~/.ssh/authorized_keys"

Write-Host "Installing SSH key on $target. Enter the remote password when prompted."
& ssh -p $ssh.Port $target $remoteCommand
if ($LASTEXITCODE -ne 0) {
    throw "ssh key installation failed with exit code $LASTEXITCODE"
}

Write-Host "Testing passwordless SSH..."
& ssh -p $ssh.Port $target "echo AIRS_SSH_KEY_OK"
if ($LASTEXITCODE -ne 0) {
    throw "passwordless SSH test failed with exit code $LASTEXITCODE"
}
