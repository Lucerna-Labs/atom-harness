using System.Diagnostics;
using System.Text.Json;

namespace LucernaLabs.AtomHarness.Desktop;

internal sealed record BackendStartup(
    string Runtime,
    Uri Origin,
    string OutputRoot,
    string WikiRuntime,
    string RagRuntime,
    string SideViewRuntime,
    string ArtifactBindingMarker);

internal sealed class BackendSupervisor : IAsyncDisposable
{
    private readonly DesktopPaths _paths;
    private readonly SafeDiagnostics _diagnostics;
    private readonly TaskCompletionSource<BackendStartup> _startup = new(
        TaskCreationOptions.RunContinuationsAsynchronously);
    private readonly TaskCompletionSource<int> _exit = new(
        TaskCreationOptions.RunContinuationsAsynchronously);
    private Process? _process;
    private ProcessJob? _job;
    private bool _disposed;

    internal BackendSupervisor(
        DesktopPaths paths,
        SafeDiagnostics diagnostics)
    {
        _paths = paths;
        _diagnostics = diagnostics;
    }

    internal bool IsRunning => _process is { HasExited: false };

    internal int? ProcessId => _process is { HasExited: false } process
        ? process.Id
        : null;

    internal async Task<BackendStartup> StartAsync(
        string modelPath,
        string gpuLayers,
        CancellationToken cancellationToken = default)
    {
        ObjectDisposedException.ThrowIf(_disposed, this);
        if (_process is not null)
        {
            throw new InvalidOperationException("The Atom backend is already started.");
        }

        foreach (string required in new[]
        {
            _paths.BackendExecutable,
            _paths.LlamaServerExecutable,
            modelPath,
        })
        {
            if (!File.Exists(required))
            {
                throw new FileNotFoundException(
                    "A required installed runtime file is absent.",
                    required);
            }
        }

        Directory.CreateDirectory(_paths.SessionRoot);
        ProcessStartInfo start = new(_paths.BackendExecutable)
        {
            WorkingDirectory = Path.GetDirectoryName(_paths.BackendExecutable)
                ?? _paths.InstallRoot,
            UseShellExecute = false,
            CreateNoWindow = true,
            RedirectStandardOutput = true,
            RedirectStandardError = true,
        };
        foreach (string argument in new[]
        {
            "--model-path",
            modelPath,
            "--llama-server",
            _paths.LlamaServerExecutable,
            "--gpu-layers",
            gpuLayers,
            "--max-queue-depth",
            "8",
            "--port",
            "0",
            "--output-root",
            _paths.SessionRoot,
            "--no-browser",
        })
        {
            start.ArgumentList.Add(argument);
        }

        start.Environment["NO_PROXY"] = "127.0.0.1,localhost";
        start.Environment["no_proxy"] = "127.0.0.1,localhost";
        Process process = new()
        {
            StartInfo = start,
            EnableRaisingEvents = true,
        };
        ProcessJob job = new();
        try
        {
            if (!process.Start())
            {
                throw new InvalidOperationException("The Atom backend did not start.");
            }

            job.Assign(process);
        }
        catch
        {
            if (!process.HasExited)
            {
                process.Kill(true);
            }

            job.Dispose();
            process.Dispose();
            throw;
        }

        _process = process;
        _job = job;
        _ = ReadStandardOutputAsync(process, cancellationToken);
        _ = ReadStandardErrorAsync(process, cancellationToken);
        _ = ObserveExitAsync(process);
        try
        {
            return await _startup.Task.WaitAsync(
                TimeSpan.FromMinutes(4),
                cancellationToken);
        }
        catch
        {
            await StopAsync(TimeSpan.FromSeconds(5));
            throw;
        }
    }

    internal async Task StopAsync(TimeSpan gracefulTimeout)
    {
        Process? process = _process;
        if (process is null || process.HasExited)
        {
            return;
        }

        try
        {
            await _exit.Task.WaitAsync(gracefulTimeout);
        }
        catch (TimeoutException)
        {
            if (!process.HasExited)
            {
                process.Kill(true);
                await process.WaitForExitAsync();
            }
        }
    }

    public async ValueTask DisposeAsync()
    {
        if (_disposed)
        {
            return;
        }

        _disposed = true;
        await StopAsync(TimeSpan.FromSeconds(5));
        _job?.Dispose();
        _job = null;
        _process?.Dispose();
        _process = null;
    }

    private async Task ReadStandardOutputAsync(
        Process process,
        CancellationToken cancellationToken)
    {
        try
        {
            while (true)
            {
                string? line = await process.StandardOutput.ReadLineAsync(
                    cancellationToken);
                if (line is null)
                {
                    return;
                }

                if (TryParseStartup(line, out BackendStartup? startup)
                    && startup is not null)
                {
                    _startup.TrySetResult(startup);
                }
                else
                {
                    await _diagnostics.RecordAsync(
                        "backend-stdout",
                        line,
                        cancellationToken);
                }
            }
        }
        catch (OperationCanceledException)
        {
            // Shutdown cancellation ends the bounded reader.
        }
        catch (Exception error)
        {
            _startup.TrySetException(error);
        }
    }

    private async Task ReadStandardErrorAsync(
        Process process,
        CancellationToken cancellationToken)
    {
        try
        {
            while (true)
            {
                string? line = await process.StandardError.ReadLineAsync(
                    cancellationToken);
                if (line is null)
                {
                    return;
                }

                await _diagnostics.RecordAsync(
                    "backend-stderr",
                    line,
                    cancellationToken);
            }
        }
        catch (OperationCanceledException)
        {
            // Shutdown cancellation ends the bounded reader.
        }
    }

    private async Task ObserveExitAsync(Process process)
    {
        await process.WaitForExitAsync();
        _exit.TrySetResult(process.ExitCode);
        if (!_startup.Task.IsCompleted)
        {
            _startup.TrySetException(
                new InvalidOperationException(
                    "The Atom backend exited before it became ready."));
        }
    }

    private static bool TryParseStartup(
        string line,
        out BackendStartup? startup)
    {
        startup = null;
        try
        {
            using JsonDocument document = JsonDocument.Parse(line);
            JsonElement root = document.RootElement;
            string runtime = root.GetProperty("runtime").GetString() ?? string.Empty;
            string originText = root.GetProperty("origin").GetString() ?? string.Empty;
            if (runtime != "atom-harness-operator-loopback-server-v1"
                || !Uri.TryCreate(originText, UriKind.Absolute, out Uri? origin)
                || origin.Scheme != Uri.UriSchemeHttp
                || origin.Host != "127.0.0.1")
            {
                return false;
            }

            startup = new BackendStartup(
                runtime,
                origin,
                root.GetProperty("output_root").GetString() ?? string.Empty,
                root.GetProperty("wiki_runtime").GetString() ?? string.Empty,
                root.GetProperty("rag_runtime").GetString() ?? string.Empty,
                root.GetProperty("side_view_runtime").GetString() ?? string.Empty,
                root.GetProperty("artifact_binding_marker").GetString()
                    ?? string.Empty);
            if (startup.WikiRuntime != "atom-language-harness-wiki-v2"
                || startup.RagRuntime != "atom-language-harness-graph-rag-v2"
                || startup.SideViewRuntime != "atom-language-harness-operator-ui-v4"
                || startup.ArtifactBindingMarker != "render_operator_surface")
            {
                startup = null;
                return false;
            }

            return true;
        }
        catch (JsonException)
        {
            return false;
        }
    }
}
