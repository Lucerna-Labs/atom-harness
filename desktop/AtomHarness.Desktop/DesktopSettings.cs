using System.Text.Json;

namespace LucernaLabs.AtomHarness.Desktop;

internal sealed class DesktopSettings
{
    internal const string Runtime = "atom-harness-desktop-settings-v1";

    public int Schema { get; init; } = 1;

    public string SettingsRuntime { get; init; } = Runtime;

    public string? ModelPath { get; set; }

    public string GpuLayers { get; set; } = "all";

    internal static DesktopSettings Load(string path)
    {
        if (!File.Exists(path))
        {
            return new DesktopSettings();
        }

        FileInfo info = new(path);
        if (info.Length is <= 0 or > 64 * 1024)
        {
            throw new InvalidDataException("Desktop settings are oversized.");
        }

        DesktopSettings settings = JsonSerializer.Deserialize<DesktopSettings>(
            File.ReadAllText(path),
            new JsonSerializerOptions
            {
                PropertyNamingPolicy = JsonNamingPolicy.SnakeCaseLower,
                PropertyNameCaseInsensitive = false,
            })
            ?? throw new InvalidDataException("Desktop settings are empty.");
        if (settings.Schema != 1
            || settings.SettingsRuntime != Runtime
            || settings.GpuLayers is not ("all" or "auto")
            || (settings.ModelPath is not null
                && !Path.IsPathFullyQualified(settings.ModelPath)))
        {
            throw new InvalidDataException("Desktop settings are invalid.");
        }

        return settings;
    }

    internal async Task SaveAsync(
        string path,
        CancellationToken cancellationToken = default)
    {
        string directory = Path.GetDirectoryName(path)
            ?? throw new InvalidDataException("The settings path is invalid.");
        Directory.CreateDirectory(directory);
        string temporary = path + ".tmp-" + Guid.NewGuid().ToString("N");
        string payload = JsonSerializer.Serialize(
            this,
            new JsonSerializerOptions
            {
                PropertyNamingPolicy = JsonNamingPolicy.SnakeCaseLower,
                WriteIndented = true,
            });
        await File.WriteAllTextAsync(temporary, payload, cancellationToken);
        File.Move(temporary, path, true);
    }
}
