using System.Text.Json;

namespace LucernaLabs.AtomHarness.Desktop;

public sealed class ReleaseManifest
{
    public int Schema { get; init; }

    public string Runtime { get; init; } = string.Empty;

    public string Version { get; init; } = string.Empty;

    public IReadOnlyList<ReleaseFile> Files { get; init; } = [];

    public static ReleaseManifest Load(string path)
    {
        ReleaseManifest manifest = JsonSerializer.Deserialize<ReleaseManifest>(
            File.ReadAllText(path),
            JsonConfiguration.Options)
            ?? throw new InvalidDataException("The release manifest is empty.");
        manifest.Validate();
        return manifest;
    }

    public void Validate()
    {
        if (Schema != 1 || Runtime != "atom-harness-release-manifest-v1")
        {
            throw new InvalidDataException("The release manifest identity is invalid.");
        }

        _ = StableVersion.Parse(Version);
        if (Files.Count == 0 || Files.Count > 10_000)
        {
            throw new InvalidDataException("The release file count is invalid.");
        }

        HashSet<string> paths = new(StringComparer.OrdinalIgnoreCase);
        foreach (ReleaseFile file in Files)
        {
            string normalized = file.Path.Replace('\\', '/');
            if (!Integrity.IsSafeRelativePath(file.Path)
                || !paths.Add(normalized)
                || file.Bytes < 0
                || !Integrity.IsSha256(file.Sha256))
            {
                throw new InvalidDataException(
                    "The release manifest contains an unsafe file.");
            }
        }
    }

    public async Task VerifyDirectoryAsync(
        string root,
        CancellationToken cancellationToken = default)
    {
        Validate();
        string fullRoot = Path.GetFullPath(root);
        foreach (ReleaseFile file in Files)
        {
            cancellationToken.ThrowIfCancellationRequested();
            string path = Path.GetFullPath(
                Path.Combine(fullRoot, file.Path.Replace('/', Path.DirectorySeparatorChar)));
            if (!Integrity.IsWithin(path, fullRoot)
                || !File.Exists(path)
                || new FileInfo(path).Length != file.Bytes)
            {
                throw new InvalidDataException(
                    $"Release file verification failed: {file.Path}");
            }

            string digest = await Integrity.Sha256FileAsync(path, cancellationToken);
            if (!string.Equals(digest, file.Sha256, StringComparison.Ordinal))
            {
                throw new InvalidDataException(
                    $"Release file SHA-256 failed: {file.Path}");
            }
        }
    }
}

public sealed class ReleaseFile
{
    public string Path { get; init; } = string.Empty;

    public long Bytes { get; init; }

    public string Sha256 { get; init; } = string.Empty;
}
