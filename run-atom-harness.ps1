param(
    [Parameter(Mandatory = $true)]
    [string]$Question,

    [ValidateSet('llama-cpp', 'openrouter')]
    [string]$Provider = $env:ATOM_LLM_PROVIDER,

    [string]$ModelPath = $env:ATOM_LLM_MODEL_PATH,

    [string]$LlmModel = $env:ATOM_LLM_MODEL,

    [string]$OutputDir,

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
if (-not $Provider) {
    $Provider = if ($env:OPENROUTER_API_KEY) {
        'openrouter'
    } else {
        'llama-cpp'
    }
}
if ($Provider -eq 'llama-cpp' -and -not $ModelPath) {
    throw 'Set ATOM_LLM_MODEL_PATH or pass -ModelPath.'
}
if (-not $LlmModel) {
    $LlmModel = 'mistralai/mistral-small-3.2-24b-instruct'
}

$arguments = @(
    (Join-Path $projectRoot 'atom_harness_experiment.py'),
    '--question', $Question,
    '--provider', $Provider,
    '--llm-model', $LlmModel,
    '--llama-cli', $LlamaCli
)
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
