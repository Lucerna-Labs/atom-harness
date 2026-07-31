using System.Security.Cryptography;
using System.Text;
using System.Text.Json;

namespace LucernaLabs.AtomHarness.Desktop;

internal sealed class SafeDiagnostics(string logPath)
{
    private const long MaximumLogBytes = 1024 * 1024;
    private readonly SemaphoreSlim _writeLock = new(1, 1);

    internal string ErrorIdentity(Exception error)
    {
        return Identity(error.GetType().FullName + "\n" + error.Message);
    }

    internal async Task RecordAsync(
        string category,
        string value,
        CancellationToken cancellationToken = default)
    {
        await _writeLock.WaitAsync(cancellationToken);
        try
        {
            Directory.CreateDirectory(
                Path.GetDirectoryName(logPath)
                ?? throw new InvalidDataException("The diagnostics path is invalid."));
            if (File.Exists(logPath)
                && new FileInfo(logPath).Length >= MaximumLogBytes)
            {
                return;
            }

            string line = JsonSerializer.Serialize(new
            {
                timestamp_utc = DateTime.UtcNow,
                category,
                sha256 = Identity(value),
            });
            await File.AppendAllTextAsync(
                logPath,
                line + Environment.NewLine,
                cancellationToken);
        }
        finally
        {
            _writeLock.Release();
        }
    }

    private static string Identity(string value)
    {
        byte[] digest = SHA256.HashData(Encoding.UTF8.GetBytes(value));
        return Convert.ToHexStringLower(digest);
    }
}
