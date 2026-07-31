using System.Diagnostics;
using System.Security.Cryptography;
using System.Text;
using System.Text.Json;

namespace LucernaLabs.AtomHarness.Desktop;

internal static class Program
{
    private static async Task<int> Main(string[] arguments)
    {
        IReadOnlyDictionary<string, string> values;
        try
        {
            values = ParseArguments(arguments);
            UpdateApplyRequest request = new(
                Required(values, "--package"),
                Required(values, "--sha256"),
                Required(values, "--install-dir"),
                Required(values, "--staging-root"),
                int.Parse(
                    Required(values, "--wait-pid"),
                    System.Globalization.CultureInfo.InvariantCulture),
                values.GetValueOrDefault("--restart"));
            UpdateApplyResult result = await SafeUpdateInstaller.ApplyAsync(request);
            await WriteResultAsync(
                request.StagingRoot,
                new
                {
                    schema = 1,
                    runtime = "lucerna-safe-updater-v1",
                    passed = true,
                    result.InstallDirectory,
                    result.BackupDirectory,
                    result.ReceiptPath,
                    completed_at_utc = DateTime.UtcNow,
                });
            return 0;
        }
        catch (Exception error)
        {
            string? stagingRoot = null;
            string? installDirectory = null;
            int waitProcessId = 0;
            string? restart = null;
            try
            {
                values = ParseArguments(arguments);
                stagingRoot = values.GetValueOrDefault("--staging-root");
                installDirectory = values.GetValueOrDefault("--install-dir");
                restart = values.GetValueOrDefault("--restart");
                _ = int.TryParse(
                    values.GetValueOrDefault("--wait-pid"),
                    out waitProcessId);
                if (!string.IsNullOrWhiteSpace(stagingRoot))
                {
                    await WriteResultAsync(
                        stagingRoot,
                        new
                        {
                            schema = 1,
                            runtime = "lucerna-safe-updater-v1",
                            passed = false,
                            error_sha256 = ErrorIdentity(error),
                            completed_at_utc = DateTime.UtcNow,
                        });
                }

                await WaitForExitAsync(waitProcessId);
                if (!string.IsNullOrWhiteSpace(installDirectory)
                    && !string.IsNullOrWhiteSpace(restart))
                {
                    string restartPath = Path.GetFullPath(
                        Path.Combine(installDirectory, restart));
                    if (Integrity.IsWithin(restartPath, installDirectory)
                        && File.Exists(restartPath))
                    {
                        Process.Start(new ProcessStartInfo(restartPath)
                        {
                            UseShellExecute = true,
                            WorkingDirectory = installDirectory,
                        });
                    }
                }
            }
            catch
            {
                // A second failure cannot weaken the original fail-closed result.
            }

            return 1;
        }
    }

    private static IReadOnlyDictionary<string, string> ParseArguments(
        IReadOnlyList<string> arguments)
    {
        if (arguments.Count == 0 || arguments.Count % 2 != 0)
        {
            throw new ArgumentException("Updater arguments must be key-value pairs.");
        }

        Dictionary<string, string> values = new(StringComparer.Ordinal);
        for (int index = 0; index < arguments.Count; index += 2)
        {
            string key = arguments[index];
            if (!key.StartsWith("--", StringComparison.Ordinal)
                || !values.TryAdd(key, arguments[index + 1]))
            {
                throw new ArgumentException("Updater arguments are invalid.");
            }
        }

        return values;
    }

    private static string Required(
        IReadOnlyDictionary<string, string> values,
        string key)
    {
        if (!values.TryGetValue(key, out string? value)
            || string.IsNullOrWhiteSpace(value))
        {
            throw new ArgumentException($"Missing updater argument: {key}");
        }

        return value;
    }

    private static async Task WriteResultAsync(string root, object result)
    {
        string fullRoot = Path.GetFullPath(root);
        Directory.CreateDirectory(fullRoot);
        string path = Path.Combine(fullRoot, "updater-result.json");
        string temporary = path + ".tmp-" + Guid.NewGuid().ToString("N");
        await File.WriteAllTextAsync(
            temporary,
            JsonSerializer.Serialize(
                result,
                new JsonSerializerOptions
                {
                    PropertyNamingPolicy = JsonNamingPolicy.SnakeCaseLower,
                    WriteIndented = true,
                }));
        File.Move(temporary, path, true);
    }

    private static async Task WaitForExitAsync(int processId)
    {
        if (processId <= 0)
        {
            return;
        }

        try
        {
            using Process process = Process.GetProcessById(processId);
            await process.WaitForExitAsync();
        }
        catch (ArgumentException)
        {
            // The application already exited.
        }
    }

    private static string ErrorIdentity(Exception error)
    {
        byte[] digest = SHA256.HashData(
            Encoding.UTF8.GetBytes(error.GetType().FullName + "\n" + error.Message));
        return Convert.ToHexStringLower(digest);
    }
}
