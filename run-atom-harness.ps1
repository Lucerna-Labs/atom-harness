param(
    [Parameter(Mandatory = $true)]
    [string]$Question,

    [ValidateSet('llama-cpp', 'openrouter')]
    [string]$Provider = $env:ATOM_LLM_PROVIDER,

    [string]$ProviderChain = $env:ATOM_LLM_PROVIDERS,

    [string]$ModelPath = $env:ATOM_LLM_MODEL_PATH,

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

    [ValidateRange(1024, 131072)]
    [int]$ContextLength = 8192,

    [string]$GpuLayers = $(if ($env:ATOM_LLM_GPU_LAYERS) {
        $env:ATOM_LLM_GPU_LAYERS
    } else {
        'auto'
    }),

    [string]$LlamaCli = $(if ($env:ATOM_LLAMA_CLI) {
        $env:ATOM_LLAMA_CLI
    } else {
        'llama-cli'
    })
)

$ErrorActionPreference = 'Stop'
$projectRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
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
    } elseif ($ModelPath -and $env:OPENROUTER_API_KEY) {
        'llama-cpp,openrouter'
    } elseif ($ModelPath) {
        'llama-cpp'
    } else {
        'openrouter'
    }
}
if ($env:ATOM_ALLOW_CLOUD_DATA -eq '1') {
    $AllowCloud = $true
}
if (-not $LlmModel) {
    $LlmModel = 'mistralai/mistral-small-3.2-24b-instruct'
}

$arguments = @(
    (Join-Path $projectRoot 'atom_harness_experiment.py'),
    '--question', $Question,
    '--providers', $ProviderChain,
    '--llm-model', $LlmModel,
    '--llama-cli', $LlamaCli,
    '--max-provider-retries', $MaxProviderRetries,
    '--retry-backoff-seconds', $RetryBackoffSeconds,
    '--circuit-failure-threshold', $CircuitFailureThreshold,
    '--circuit-cooldown-seconds', $CircuitCooldownSeconds,
    '--max-concurrency', $MaxConcurrency,
    '--acquire-timeout-seconds', $AcquireTimeoutSeconds,
    '--provider-timeout-seconds', $ProviderTimeoutSeconds,
    '--context-length', $ContextLength,
    '--gpu-layers', $GpuLayers
)
if ($AllowCloud) {
    $arguments += '--allow-cloud'
}
if ($ModelPath) {
    $arguments += @('--model-path', $ModelPath)
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
