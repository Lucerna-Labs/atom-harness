param(
    [string[]]$Question,

    [string]$QuestionsFile,

    [string]$OutputRoot,

    [string]$ModelPath = $env:ATOM_LLM_MODEL_PATH,

    [Alias('LlamaCompletion', 'LlamaCli')]
    [string]$LlamaServer = $(if ($env:ATOM_LLAMA_SERVER) {
        $env:ATOM_LLAMA_SERVER
    } elseif ($env:ATOM_LLAMA_COMPLETION) {
        $env:ATOM_LLAMA_COMPLETION
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
    [int]$LaneStartupTimeoutSeconds = 0,

    [ValidateRange(0, 3600)]
    [double]$LaneAcquireTimeoutSeconds = 0,

    [ValidateRange(0, 16)]
    [int]$LaneParallelSlots = 0,

    [ValidateRange(0, 256)]
    [int]$LaneMaxQueueDepth = 0,

    [ValidateRange(1, 64)]
    [int]$MaxConcurrency = 2
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
    $residentLane.automatic_restart_on_next_request -ne $true -or
    $residentLane.web_ui_enabled -ne $false
) {
    throw 'The Atom resident language-lane contract is invalid.'
}
if (-not $Question -and -not $QuestionsFile) {
    throw 'Provide at least one -Question or a -QuestionsFile.'
}
if ($Question.Count -gt 256) {
    throw 'A resident session accepts no more than 256 direct questions.'
}
if ($QuestionsFile -and -not (
    Test-Path -LiteralPath $QuestionsFile -PathType Leaf
)) {
    throw 'The resident session questions file is absent.'
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
if ($LaneStartupTimeoutSeconds -eq 0) {
    $LaneStartupTimeoutSeconds = [int]$residentLane.startup_timeout_seconds
}
if ($LaneAcquireTimeoutSeconds -eq 0) {
    $LaneAcquireTimeoutSeconds = [double]$residentLane.acquire_timeout_seconds
}
if ($LaneParallelSlots -eq 0) {
    $LaneParallelSlots = [int]$residentLane.parallel_slots
}
if ($PSBoundParameters.ContainsKey('LaneMaxQueueDepth') -eq $false) {
    $LaneMaxQueueDepth = [int]$residentLane.max_queue_depth
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

$arguments = @(
    (Join-Path $projectRoot 'atom_harness_session_cli.py'),
    '--model-path', $ModelPath,
    '--llama-server', $LlamaServer,
    '--gpu-layers', $GpuLayers,
    '--provider-timeout-seconds', $ProviderTimeoutSeconds,
    '--lane-startup-timeout-seconds', $LaneStartupTimeoutSeconds,
    '--lane-acquire-timeout-seconds', $LaneAcquireTimeoutSeconds,
    '--lane-parallel-slots', $LaneParallelSlots,
    '--lane-max-queue-depth', $LaneMaxQueueDepth,
    '--max-concurrency', $MaxConcurrency
)
foreach ($item in $Question) {
    $arguments += @('--question', $item)
}
if ($QuestionsFile) {
    $arguments += @(
        '--questions-file',
        [System.IO.Path]::GetFullPath($QuestionsFile)
    )
}
if ($OutputRoot) {
    $arguments += @('--output-root', $OutputRoot)
}

Push-Location $projectRoot
try {
    & $pythonExecutable @arguments
    exit $LASTEXITCODE
} finally {
    Pop-Location
}
