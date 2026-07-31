param(
    [ValidatePattern('^\d+\.\d+\.\d+$')]
    [string]$Version = '5.0.0',

    [string]$OutputRoot,

    [string]$LlamaServer = $env:ATOM_LLAMA_SERVER,

    [switch]$SkipMsi
)

$ErrorActionPreference = 'Stop'

function Get-PortableRelativePath {
    param(
        [Parameter(Mandatory = $true)]
        [string]$Root,

        [Parameter(Mandatory = $true)]
        [string]$Path
    )

    $rootPath = [System.IO.Path]::GetFullPath($Root)
    if (-not $rootPath.EndsWith(
        [System.IO.Path]::DirectorySeparatorChar.ToString()
    )) {
        $rootPath += [System.IO.Path]::DirectorySeparatorChar
    }
    $rootUri = [System.Uri]::new($rootPath)
    $pathUri = [System.Uri]::new([System.IO.Path]::GetFullPath($Path))
    return [System.Uri]::UnescapeDataString(
        $rootUri.MakeRelativeUri($pathUri).ToString()
    )
}

function Set-HarvestComponentGuids {
    param(
        [Parameter(Mandatory = $true)]
        [string]$WixSource
    )

    [xml]$document = Get-Content -LiteralPath $WixSource -Raw
    $namespaceManager = [System.Xml.XmlNamespaceManager]::new(
        $document.NameTable
    )
    $namespaceManager.AddNamespace(
        'wix',
        'http://schemas.microsoft.com/wix/2006/wi'
    )
    $components = $document.SelectNodes(
        '//wix:Component',
        $namespaceManager
    )
    $sha256 = [System.Security.Cryptography.SHA256]::Create()
    try {
        foreach ($component in $components) {
            $identity = (
                'com.lucernalabs.atom-harness.wix-component:' +
                $component.Id
            )
            $digest = $sha256.ComputeHash(
                [System.Text.Encoding]::UTF8.GetBytes($identity)
            )
            $guidBytes = [byte[]]::new(16)
            [System.Array]::Copy($digest, $guidBytes, 16)
            $component.SetAttribute(
                'Guid',
                ([System.Guid]::new($guidBytes)).ToString('B')
            )
        }
    }
    finally {
        $sha256.Dispose()
    }

    $settings = [System.Xml.XmlWriterSettings]::new()
    $settings.Encoding = [System.Text.UTF8Encoding]::new($false)
    $settings.Indent = $true
    $writer = [System.Xml.XmlWriter]::Create($WixSource, $settings)
    try {
        $document.Save($writer)
    }
    finally {
        $writer.Dispose()
    }
}

$projectRoot = Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path)
$stamp = Get-Date -Format 'yyyyMMdd-HHmmss'
if (-not $OutputRoot) {
    $OutputRoot = Join-Path $projectRoot "local-results\desktop-v5-package-$stamp"
}
$OutputRoot = [System.IO.Path]::GetFullPath($OutputRoot)
$localResults = [System.IO.Path]::GetFullPath(
    (Join-Path $projectRoot 'local-results')
)
if (-not $OutputRoot.StartsWith(
    $localResults + [System.IO.Path]::DirectorySeparatorChar,
    [System.StringComparison]::OrdinalIgnoreCase
)) {
    throw 'Desktop package output must stay below local-results.'
}
if (Test-Path -LiteralPath $OutputRoot) {
    throw 'Desktop package output already exists. Choose a new output directory.'
}

$updateContract = Get-Content -LiteralPath (
    Join-Path $projectRoot 'lucerna-update.json'
) -Raw | ConvertFrom-Json
if (
    $updateContract.schema -ne 1 -or
    $updateContract.current_version -ne $Version -or
    $updateContract.policy.explicit_user_consent_required -ne $true -or
    $updateContract.policy.artifact_sha256_required -ne $true -or
    $updateContract.policy.stage_outside_install_directory -ne $true -or
    $updateContract.policy.replace_only_after_app_exit -ne $true
) {
    throw 'The opt-in update contract is invalid for this package.'
}

if (-not $LlamaServer) {
    $llamaCommand = Get-Command llama-server -ErrorAction SilentlyContinue
    if ($llamaCommand) {
        $LlamaServer = $llamaCommand.Source
    }
}
if (-not $LlamaServer -or -not (
    Test-Path -LiteralPath $LlamaServer -PathType Leaf
)) {
    throw 'A verified llama-server runtime is required for desktop packaging.'
}
$LlamaServer = [System.IO.Path]::GetFullPath($LlamaServer)
$llamaRoot = Split-Path -Parent $LlamaServer

$python = Join-Path $env:LOCALAPPDATA 'Programs\Python\Python313\python.exe'
if (-not (Test-Path -LiteralPath $python -PathType Leaf)) {
    throw 'Python 3.13 is required for desktop packaging.'
}
$pyInstallerVersion = & $python -m PyInstaller --version
if ($LASTEXITCODE -ne 0 -or $pyInstallerVersion.Trim() -ne '6.21.0') {
    throw 'PyInstaller 6.21.0 is required for desktop packaging.'
}

New-Item -ItemType Directory -Path $OutputRoot | Out-Null
$buildRoot = Join-Path $OutputRoot 'build'
$stageRoot = Join-Path $OutputRoot 'stage'
$appRoot = Join-Path $stageRoot 'app'
$backendDist = Join-Path $buildRoot 'backend-dist'
$backendWork = Join-Path $buildRoot 'backend-work'
$desktopPublish = Join-Path $buildRoot 'desktop-publish'
$updaterPublish = Join-Path $buildRoot 'updater-publish'
foreach ($directory in @(
    $buildRoot,
    $stageRoot,
    $appRoot,
    $backendDist,
    $backendWork,
    $desktopPublish,
    $updaterPublish
)) {
    New-Item -ItemType Directory -Path $directory | Out-Null
}

Push-Location $projectRoot
try {
    & $python scripts\verify_atom_harness_v5.py
    if ($LASTEXITCODE -ne 0) {
        throw 'Operator V5 repository policy failed before packaging.'
    }

    foreach ($project in @(
        'desktop\AtomHarness.Desktop.Core\AtomHarness.Desktop.Core.csproj',
        'desktop\AtomHarness.Desktop\AtomHarness.Desktop.csproj',
        'desktop\AtomHarness.Updater\AtomHarness.Updater.csproj',
        'desktop\AtomHarness.Desktop.Tests\AtomHarness.Desktop.Tests.csproj'
    )) {
        & dotnet restore $project --locked-mode
        if ($LASTEXITCODE -ne 0) {
            throw "Locked .NET restore failed: $project"
        }
    }

    & cargo build `
        --locked `
        --release `
        --manifest-path atom_causal_memory_rust\Cargo.toml `
        -p atom-causal-memory
    if ($LASTEXITCODE -ne 0) {
        throw 'The bundled causal-memory runtime build failed.'
    }

    & dotnet test `
        desktop\AtomHarness.Desktop.Tests\AtomHarness.Desktop.Tests.csproj `
        --configuration Release `
        --no-restore
    if ($LASTEXITCODE -ne 0) {
        throw 'Desktop .NET tests failed.'
    }

    & $python -m PyInstaller `
        --noconfirm `
        --clean `
        --distpath $backendDist `
        --workpath $backendWork `
        atom-harness-backend.spec
    if ($LASTEXITCODE -ne 0) {
        throw 'The frozen Atom backend build failed.'
    }

    & dotnet publish `
        desktop\AtomHarness.Desktop\AtomHarness.Desktop.csproj `
        --configuration Release `
        --runtime win-x64 `
        --self-contained true `
        --no-restore `
        --output $desktopPublish
    if ($LASTEXITCODE -ne 0) {
        throw 'The desktop shell publish failed.'
    }

    & dotnet publish `
        desktop\AtomHarness.Updater\AtomHarness.Updater.csproj `
        --configuration Release `
        --runtime win-x64 `
        --self-contained true `
        --no-restore `
        --output $updaterPublish
    if ($LASTEXITCODE -ne 0) {
        throw 'The update helper publish failed.'
    }
} finally {
    Pop-Location
}

Get-ChildItem -LiteralPath $desktopPublish -Force | ForEach-Object {
    Copy-Item -LiteralPath $_.FullName -Destination $appRoot -Recurse
}
$toolsRoot = Join-Path $appRoot 'tools'
New-Item -ItemType Directory -Path $toolsRoot | Out-Null
Copy-Item -LiteralPath (
    Join-Path $updaterPublish 'AtomHarness.Updater.exe'
) -Destination $toolsRoot

$runtimeRoot = Join-Path $appRoot 'runtime'
$backendRoot = Join-Path $runtimeRoot 'backend'
$llamaStageRoot = Join-Path $runtimeRoot 'llama'
New-Item -ItemType Directory -Path $runtimeRoot | Out-Null
Copy-Item -LiteralPath (
    Join-Path $backendDist 'atom-harness-backend'
) -Destination $backendRoot -Recurse
New-Item -ItemType Directory -Path $llamaStageRoot | Out-Null
Get-ChildItem -LiteralPath $llamaRoot -File |
    Where-Object {
        $_.Name -eq 'llama-server.exe' -or $_.Extension -eq '.dll'
    } |
    ForEach-Object {
        Copy-Item -LiteralPath $_.FullName -Destination $llamaStageRoot
    }

foreach ($contract in @(
    'atom-language-model.json',
    'atom-harness-desktop-architecture.json',
    'lucerna-update.json'
)) {
    Copy-Item -LiteralPath (Join-Path $projectRoot $contract) -Destination $appRoot
}

$releaseFiles = @(
    Get-ChildItem -LiteralPath $appRoot -File -Recurse |
        Sort-Object FullName |
        ForEach-Object {
            $relative = Get-PortableRelativePath -Root $appRoot -Path $_.FullName
            [ordered]@{
                path = $relative
                bytes = $_.Length
                sha256 = (
                    Get-FileHash -LiteralPath $_.FullName -Algorithm SHA256
                ).Hash.ToLowerInvariant()
            }
        }
)
$releaseManifest = [ordered]@{
    schema = 1
    runtime = 'atom-harness-release-manifest-v1'
    version = $Version
    files = $releaseFiles
}
$releaseManifestPath = Join-Path $appRoot (
    'atom-harness-release-manifest.json'
)
[System.IO.File]::WriteAllText(
    $releaseManifestPath,
    ($releaseManifest | ConvertTo-Json -Depth 8),
    [System.Text.UTF8Encoding]::new($false)
)

$zipPath = Join-Path $OutputRoot "Atom-Harness-$Version-windows-x64.zip"
Compress-Archive -LiteralPath $appRoot -DestinationPath $zipPath `
    -CompressionLevel Optimal
$zipInfo = Get-Item -LiteralPath $zipPath
$zipSha256 = (
    Get-FileHash -LiteralPath $zipPath -Algorithm SHA256
).Hash.ToLowerInvariant()
$feed = [ordered]@{
    schema = 1
    app_id = 'com.lucernalabs.atom-harness'
    platform = 'windows-x64'
    version = $Version
    release_notes = 'Atom Harness Desktop Phase 5.'
    artifact = [ordered]@{
        url = (
            "https://github.com/Lucerna-Labs/atom-harness/releases/" +
            "download/v$Version/Atom-Harness-$Version-windows-x64.zip"
        )
        bytes = $zipInfo.Length
        sha256 = $zipSha256
    }
}
$feedPath = Join-Path $OutputRoot 'lucerna-update-feed.json'
[System.IO.File]::WriteAllText(
    $feedPath,
    ($feed | ConvertTo-Json -Depth 6),
    [System.Text.UTF8Encoding]::new($false)
)

$msiPath = $null
if (-not $SkipMsi) {
    $wixRoot = Join-Path $buildRoot 'wix'
    New-Item -ItemType Directory -Path $wixRoot | Out-Null
    $fragmentPath = Join-Path $wixRoot 'AppFiles.wxs'
    $harvestTransform = Join-Path $projectRoot (
        'desktop\packaging\PerUserHarvest.xslt'
    )
    $heat = 'C:\Program Files (x86)\WiX Toolset v3.14\bin\heat.exe'
    $candle = 'C:\Program Files (x86)\WiX Toolset v3.14\bin\candle.exe'
    $light = 'C:\Program Files (x86)\WiX Toolset v3.14\bin\light.exe'
    foreach ($tool in @($heat, $candle, $light)) {
        if (-not (Test-Path -LiteralPath $tool -PathType Leaf)) {
            throw 'WiX Toolset 3.14 is required for MSI packaging.'
        }
    }

    & $heat dir $appRoot `
        -nologo `
        -cg AppFiles `
        -dr INSTALLFOLDER `
        -srd `
        -sfrag `
        -scom `
        -sreg `
        -ag `
        -t $harvestTransform `
        -var var.SourceDir `
        -out $fragmentPath
    if ($LASTEXITCODE -ne 0) {
        throw 'WiX directory harvesting failed.'
    }
    Set-HarvestComponentGuids -WixSource $fragmentPath

    $productWxs = Join-Path $projectRoot (
        'desktop\packaging\AtomHarness.wxs'
    )
    & $candle `
        -nologo `
        -wx `
        "-dSourceDir=$appRoot" `
        "-dProductVersion=$Version" `
        -out ($wixRoot + '\') `
        $productWxs `
        $fragmentPath
    if ($LASTEXITCODE -ne 0) {
        throw 'WiX source compilation failed.'
    }

    $msiPath = Join-Path $OutputRoot "Atom-Harness-$Version-windows-x64.msi"
    & $light `
        -nologo `
        -wx `
        -sice:ICE91 `
        -out $msiPath `
        (Join-Path $wixRoot 'AtomHarness.wixobj') `
        (Join-Path $wixRoot 'AppFiles.wixobj')
    if ($LASTEXITCODE -ne 0) {
        throw 'WiX MSI linking failed.'
    }
}

$msiBytes = $null
$msiSha256 = $null
if ($msiPath) {
    $msiBytes = (Get-Item -LiteralPath $msiPath).Length
    $msiSha256 = (
        Get-FileHash -LiteralPath $msiPath -Algorithm SHA256
    ).Hash.ToLowerInvariant()
}
$result = [ordered]@{
    schema = 1
    runtime = 'atom-harness-desktop-package-result-v1'
    passed = $true
    version = $Version
    app_root = $appRoot
    portable_zip = $zipPath
    portable_zip_bytes = $zipInfo.Length
    portable_zip_sha256 = $zipSha256
    update_feed = $feedPath
    msi = $msiPath
    msi_bytes = $msiBytes
    msi_sha256 = $msiSha256
    llama_server_sha256 = (
        Get-FileHash -LiteralPath (
            Join-Path $llamaStageRoot 'llama-server.exe'
        ) -Algorithm SHA256
    ).Hash.ToLowerInvariant()
    file_count = $releaseFiles.Count
    completed_at_utc = [DateTime]::UtcNow.ToString('o')
}
$resultPath = Join-Path $OutputRoot 'atom-harness-desktop-package.json'
[System.IO.File]::WriteAllText(
    $resultPath,
    ($result | ConvertTo-Json -Depth 6),
    [System.Text.UTF8Encoding]::new($false)
)
$result | ConvertTo-Json -Depth 6
