param(
    [string]$OutputRoot,

    [string]$ModelPath = $env:ATOM_LLM_MODEL_PATH,

    [Alias('LlamaCompletion', 'LlamaCli')]
    [string]$LlamaServer = $(if ($env:ATOM_LLAMA_SERVER) {
        $env:ATOM_LLAMA_SERVER
    } else {
        ''
    }),

    [string]$GpuLayers = $(if ($env:ATOM_LLM_GPU_LAYERS) {
        $env:ATOM_LLM_GPU_LAYERS
    } else {
        'auto'
    }),

    [ValidateRange(1, 3600)]
    [int]$ProviderTimeoutSeconds = 240,

    [ValidateRange(0, 3600)]
    [int]$StartupTimeoutSeconds = 0,

    [ValidateRange(0, 3600)]
    [double]$LaneAcquireTimeoutSeconds = 0,

    [ValidateRange(1, 256)]
    [int]$MaxQueueDepth = 8,

    [string]$ToolWorkspace = $(if ($env:ATOM_TOOL_WORKSPACE) {
        $env:ATOM_TOOL_WORKSPACE
    } else {
        'C:\Projects'
    }),

    [ValidateRange(0, 65535)]
    [int]$Port = $(if ($env:ATOM_HARNESS_OPERATOR_PORT) {
        [int]$env:ATOM_HARNESS_OPERATOR_PORT
    } else {
        0
    }),

    [switch]$NoBrowser
)

$ErrorActionPreference = 'Stop'
$projectRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$contractPath = Join-Path $projectRoot 'atom-language-model.json'
$registryPath = Join-Path $projectRoot 'ai-runtime-registry.json'

try {
    $modelContract = Get-Content -LiteralPath $contractPath -Raw |
        ConvertFrom-Json -ErrorAction Stop
    $runtimeRegistry = Get-Content -LiteralPath $registryPath -Raw |
        ConvertFrom-Json -ErrorAction Stop
} catch {
    throw "The Atom operator contracts are unavailable: $($_.Exception.Message)"
}

if (
    $runtimeRegistry.schema_version -ne 1 -or
    $runtimeRegistry.active_runtime -ne 'language-harness-v6' -or
    $runtimeRegistry.runtimes.'language-harness-v6'.runtime_entrypoint -ne
        'atom_harness_operator_server.py' -or
    $runtimeRegistry.runtimes.'language-harness-v6'.artifact_binding_marker -ne
        'render_operator_surface' -or
    $runtimeRegistry.runtimes.'language-harness-v6'.tool_artifact_binding_marker -ne
        'render_atom_tool_artifact'
) {
    throw 'The active Atom Harness Operator V6 registry is invalid.'
}

if (
    $modelContract.schema -ne 1 -or
    $modelContract.runtime -ne 'atom-language-model-contract-v1' -or
    $modelContract.default_provider -ne 'llama-cpp' -or
    $modelContract.runtime_policy.executable -ne 'llama-server'
) {
    throw 'The Atom resident language-model contract identity is invalid.'
}

$residentLane = $modelContract.runtime_policy.resident_lane
if (
    $residentLane.runtime -ne 'atom-resident-language-lane-v1' -or
    $residentLane.topology -ne 'spiderweb-permanent-elevated-language-lane' -or
    $residentLane.host -ne '127.0.0.1' -or
    $residentLane.api_key_in_memory_only -ne $true -or
    $residentLane.external_proxy_disabled -ne $true -or
    $residentLane.preload_inference_path -ne $true -or
    $residentLane.web_ui_enabled -ne $false
) {
    throw 'The Atom resident language-lane contract is invalid.'
}

if (-not $ModelPath) {
    $ModelPath = [System.IO.Path]::GetFullPath(
        (Join-Path $projectRoot $modelContract.artifact.default_relative_path)
    )
}
if (-not (Test-Path -LiteralPath $ModelPath -PathType Leaf)) {
    throw 'The local Atom language model is absent. Run .\install-atom-language-model.ps1 first.'
}
if (-not $LlamaServer) {
    $LlamaServer = [string]$modelContract.runtime_policy.executable
}
$serverCommand = Get-Command $LlamaServer -ErrorAction SilentlyContinue
if (-not $serverCommand -and -not (
    Test-Path -LiteralPath $LlamaServer -PathType Leaf
)) {
    throw 'llama-server is absent from PATH and the configured path.'
}
if ($StartupTimeoutSeconds -eq 0) {
    $StartupTimeoutSeconds = [int]$residentLane.startup_timeout_seconds
}
if ($LaneAcquireTimeoutSeconds -eq 0) {
    $LaneAcquireTimeoutSeconds = [double]$residentLane.acquire_timeout_seconds
}
if (-not $PSBoundParameters.ContainsKey('MaxQueueDepth')) {
    $MaxQueueDepth = [int]$residentLane.max_queue_depth
}

$pythonCandidates = @(
    (Join-Path $env:LOCALAPPDATA 'Programs\Python\Python313\python.exe'),
    (Join-Path $env:LOCALAPPDATA 'Programs\Python\Python312\python.exe'),
    (Join-Path $env:LOCALAPPDATA 'Programs\Python\Python311\python.exe')
)
$pythonExecutable = $null
foreach ($candidate in $pythonCandidates) {
    if (-not (Test-Path -LiteralPath $candidate -PathType Leaf)) {
        continue
    }
    & $candidate -c 'import numpy' 2>$null
    if ($LASTEXITCODE -eq 0) {
        $pythonExecutable = $candidate
        break
    }
}
if (-not $pythonExecutable) {
    $command = Get-Command python.exe -ErrorAction SilentlyContinue
    if ($command) {
        & $command.Source -c 'import numpy' 2>$null
        if ($LASTEXITCODE -eq 0) {
            $pythonExecutable = $command.Source
        }
    }
}
if (-not $pythonExecutable) {
    throw 'No Python environment with NumPy is available.'
}
if (-not (Test-Path -LiteralPath $ToolWorkspace -PathType Container)) {
    throw 'The permissioned-hands workspace must be an existing directory.'
}
$ToolWorkspace = [System.IO.Path]::GetFullPath($ToolWorkspace)

$arguments = @(
    (Join-Path $projectRoot 'atom_harness_operator_server.py'),
    '--model-path', $ModelPath,
    '--llama-server', $LlamaServer,
    '--gpu-layers', $GpuLayers,
    '--provider-timeout-seconds', $ProviderTimeoutSeconds,
    '--startup-timeout-seconds', $StartupTimeoutSeconds,
    '--lane-acquire-timeout-seconds', $LaneAcquireTimeoutSeconds,
    '--max-queue-depth', $MaxQueueDepth,
    '--tool-workspace', $ToolWorkspace,
    '--port', $Port
)
if ($OutputRoot) {
    $arguments += @('--output-root', [System.IO.Path]::GetFullPath($OutputRoot))
}
if ($NoBrowser) {
    $arguments += '--no-browser'
}

Write-Host "Preloading Atom, Qwen, and the permission registry for $ToolWorkspace."
Push-Location $projectRoot
try {
    & $pythonExecutable @arguments
    exit $LASTEXITCODE
} finally {
    Pop-Location
}
