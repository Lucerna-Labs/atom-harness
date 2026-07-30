param(
    [Parameter(Mandatory = $true)]
    [string]$Question,

    [ValidateSet('llama-cpp', 'openrouter')]
    [string]$Provider = $env:ATOM_LLM_PROVIDER,

    [string]$ProviderChain = $env:ATOM_LLM_PROVIDERS,

    [string]$ModelPath = $env:ATOM_LLM_MODEL_PATH,

    [string]$ModelSha256 = $env:ATOM_LLM_MODEL_SHA256,

    [ValidateRange(0, [long]::MaxValue)]
    [long]$ModelBytes = 0,

    [ValidateSet('qwen-chatml-manual-v1', 'raw-prompt-v1')]
    [string]$ChatTemplate = $env:ATOM_LLM_CHAT_TEMPLATE,

    [string]$LlmModel = $env:ATOM_LLM_MODEL,

    [string]$OutputDir,

    [switch]$AllowCloud,

    [ValidateRange(0, 3)]
    [int]$MaxProviderRetries = 1,

    [ValidateRange(0, 30)]
    [double]$RetryBackoffSeconds = 0.25,

    [ValidateRange(1, 10)]
    [int]$CircuitFailureThreshold = 1,

    [ValidateRange(0.1, 86400)]
    [double]$CircuitCooldownSeconds = 60,

    [ValidateRange(1, 64)]
    [int]$MaxConcurrency = 2,

    [ValidateRange(0.1, 3600)]
    [double]$AcquireTimeoutSeconds = 30,

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

    [ValidateRange(0, 131072)]
    [int]$ContextLength = 0,

    [string]$GpuLayers = $(if ($env:ATOM_LLM_GPU_LAYERS) {
        $env:ATOM_LLM_GPU_LAYERS
    } else {
        'auto'
    }),

    [Alias('LlamaCompletion', 'LlamaCli')]
    [string]$LlamaServer = $(if ($env:ATOM_LLAMA_SERVER) {
        $env:ATOM_LLAMA_SERVER
    } elseif ($env:ATOM_LLAMA_COMPLETION) {
        $env:ATOM_LLAMA_COMPLETION
    } else {
        ''
    })
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
    throw 'The Atom language-model contract identity is invalid.'
}
$officialModelName = [string]$modelContract.artifact.filename
$officialModelSha256 = [string]$modelContract.artifact.sha256
$officialModelBytes = [long]$modelContract.artifact.bytes
$officialChatTemplate = [string]$modelContract.runtime_policy.chat_template
$officialContextLength = [int]$modelContract.runtime_policy.harness_context_tokens
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
if (-not $LlamaServer) {
    $LlamaServer = [string]$modelContract.runtime_policy.executable
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
if ($ContextLength -eq 0) {
    $ContextLength = $officialContextLength
}
if ($ContextLength -lt 1024) {
    throw 'The llama.cpp context length must be at least 1024 tokens.'
}
if (-not $ModelPath) {
    $ModelPath = [System.IO.Path]::GetFullPath(
        (Join-Path $projectRoot $modelContract.artifact.default_relative_path)
    )
}
if ([System.IO.Path]::GetFileName($ModelPath) -ieq $officialModelName) {
    if (-not $ModelSha256) {
        $ModelSha256 = $officialModelSha256
    }
    if ($ModelBytes -eq 0) {
        $ModelBytes = $officialModelBytes
    }
    if (-not $ChatTemplate) {
        $ChatTemplate = $officialChatTemplate
    }
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
if (-not $ProviderChain) {
    $ProviderChain = if ($Provider) {
        $Provider
    } else {
        'llama-cpp'
    }
}
$providerNames = @(
    $ProviderChain -split ',' |
        ForEach-Object { $_.Trim().ToLowerInvariant() } |
        Where-Object { $_ }
)
if ($providerNames -contains 'llama-cpp') {
    if (-not (Test-Path -LiteralPath $ModelPath -PathType Leaf)) {
        throw 'The local Atom language model is absent. Run .\install-atom-language-model.ps1 first.'
    }
    if (-not $ModelSha256) {
        throw 'A custom local GGUF requires -ModelSha256.'
    }
    if (-not $ChatTemplate) {
        throw 'A custom local GGUF requires -ChatTemplate.'
    }
    $serverCommand = Get-Command $LlamaServer -ErrorAction SilentlyContinue
    if (-not $serverCommand -and -not (
        Test-Path -LiteralPath $LlamaServer -PathType Leaf
    )) {
        throw 'llama-server is absent from PATH and the configured path.'
    }
}
if ($env:ATOM_ALLOW_CLOUD_DATA -eq '1') {
    $AllowCloud = $true
}

$arguments = @(
    (Join-Path $projectRoot 'atom_harness_experiment.py'),
    '--question', $Question,
    '--providers', $ProviderChain,
    '--llama-server', $LlamaServer,
    '--max-provider-retries', $MaxProviderRetries,
    '--retry-backoff-seconds', $RetryBackoffSeconds,
    '--circuit-failure-threshold', $CircuitFailureThreshold,
    '--circuit-cooldown-seconds', $CircuitCooldownSeconds,
    '--max-concurrency', $MaxConcurrency,
    '--acquire-timeout-seconds', $AcquireTimeoutSeconds,
    '--provider-timeout-seconds', $ProviderTimeoutSeconds,
    '--lane-startup-timeout-seconds', $LaneStartupTimeoutSeconds,
    '--lane-acquire-timeout-seconds', $LaneAcquireTimeoutSeconds,
    '--lane-parallel-slots', $LaneParallelSlots,
    '--lane-max-queue-depth', $LaneMaxQueueDepth,
    '--context-length', $ContextLength,
    '--gpu-layers', $GpuLayers
)
if ($LlmModel) {
    $arguments += @('--llm-model', $LlmModel)
}
if ($AllowCloud) {
    $arguments += '--allow-cloud'
}
if ($ModelPath) {
    $arguments += @('--model-path', $ModelPath)
}
if ($ModelSha256) {
    $arguments += @('--model-sha256', $ModelSha256)
}
if ($ModelBytes -gt 0) {
    $arguments += @('--model-bytes', $ModelBytes)
}
if ($ChatTemplate) {
    $arguments += @('--chat-template', $ChatTemplate)
}
if ($OutputDir) {
    $arguments += @('--output-dir', $OutputDir)
}

Push-Location $projectRoot
try {
    & $pythonExecutable @arguments
    exit $LASTEXITCODE
} finally {
    Pop-Location
}
