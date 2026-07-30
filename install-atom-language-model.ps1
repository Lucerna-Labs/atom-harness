param(
    [string]$ModelDirectory,

    [ValidateRange(1, 20)]
    [int]$Retries = 5
)

$ErrorActionPreference = 'Stop'
$projectRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$contractPath = Join-Path $projectRoot 'atom-language-model.json'
try {
    $modelContract = Get-Content -LiteralPath $contractPath -Raw |
        ConvertFrom-Json -ErrorAction Stop
} catch {
    throw "The Atom language-model contract is unavailable: $($_.Exception.Message)"
}
if (
    $modelContract.schema -ne 1 -or
    $modelContract.runtime -ne 'atom-language-model-contract-v1' -or
    $modelContract.default_provider -ne 'llama-cpp'
) {
    throw 'The Atom language-model contract identity is invalid.'
}
$defaultModelPath = [System.IO.Path]::GetFullPath(
    (Join-Path $projectRoot $modelContract.artifact.default_relative_path)
)
if (-not $ModelDirectory) {
    $ModelDirectory = Split-Path -Parent $defaultModelPath
}
$modelName = [string]$modelContract.artifact.filename
$expectedSha256 = [string]$modelContract.artifact.sha256
$expectedBytes = [long]$modelContract.artifact.bytes
$repository = [string]$modelContract.artifact.repository
$downloadUrl = [string]$modelContract.artifact.download_url
$modelPath = Join-Path $ModelDirectory $modelName

function Assert-OfficialModel {
    param([Parameter(Mandatory = $true)][string]$Path)

    $file = Get-Item -LiteralPath $Path -ErrorAction Stop
    if ($file.Length -ne $expectedBytes) {
        throw (
            "Model byte count mismatch. Expected $expectedBytes, " +
            "found $($file.Length)."
        )
    }
    $actualSha256 = (
        Get-FileHash -LiteralPath $Path -Algorithm SHA256
    ).Hash.ToLowerInvariant()
    if ($actualSha256 -ne $expectedSha256) {
        throw (
            "Model SHA-256 mismatch. The file remains untrusted and " +
            'will not be admitted.'
        )
    }
    return [pscustomobject]@{
        ModelPath = $file.FullName
        Bytes = $file.Length
        Sha256 = $actualSha256
        IntegrityVerified = $true
    }
}

if (Test-Path -LiteralPath $modelPath -PathType Leaf) {
    Assert-OfficialModel -Path $modelPath
    exit 0
}

New-Item -ItemType Directory -Path $ModelDirectory -Force | Out-Null
$hf = Get-Command hf.exe -ErrorAction SilentlyContinue
if ($hf) {
    $env:HF_XET_HIGH_PERFORMANCE = '1'
    & $hf.Source download $repository $modelName --local-dir $ModelDirectory
    if ($LASTEXITCODE -ne 0) {
        throw "Hugging Face download failed with exit code $LASTEXITCODE."
    }
} else {
    $curl = Get-Command curl.exe -ErrorAction SilentlyContinue
    if (-not $curl) {
        throw 'Neither hf.exe nor curl.exe is available for the model download.'
    }
    $partialPath = "$modelPath.partial"
    & $curl.Source `
        --location `
        --fail `
        --retry $Retries `
        --retry-all-errors `
        --continue-at - `
        --output $partialPath `
        $downloadUrl
    if ($LASTEXITCODE -ne 0) {
        throw "Model download failed with exit code $LASTEXITCODE."
    }
    $verifiedPartial = Assert-OfficialModel -Path $partialPath
    if (-not $verifiedPartial.IntegrityVerified) {
        throw 'The downloaded model did not pass integrity verification.'
    }
    Move-Item -LiteralPath $partialPath -Destination $modelPath
}

if (-not (Test-Path -LiteralPath $modelPath -PathType Leaf)) {
    throw 'The downloader exited without publishing the model file.'
}
Assert-OfficialModel -Path $modelPath
