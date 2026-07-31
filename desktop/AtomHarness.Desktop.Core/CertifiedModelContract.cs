using System.Text.Json;

namespace LucernaLabs.AtomHarness.Desktop;

public sealed class CertifiedModelContract
{
    public int Schema { get; init; }

    public string Runtime { get; init; } = string.Empty;

    public string Role { get; init; } = string.Empty;

    public ModelArtifact Artifact { get; init; } = new();

    public static CertifiedModelContract Load(string path)
    {
        CertifiedModelContract contract = JsonSerializer.Deserialize<CertifiedModelContract>(
            File.ReadAllText(path),
            JsonConfiguration.Options)
            ?? throw new InvalidDataException("The model contract is empty.");
        contract.Validate();
        return contract;
    }

    public void Validate()
    {
        if (Schema != 1
            || Runtime != "atom-language-model-contract-v1"
            || Role != "language-only-membrane"
            || Artifact.Filename != "qwen3-4b-instruct-2507-q8_0.gguf"
            || Artifact.Bytes != 4_280_403_520
            || !Integrity.IsSha256(Artifact.Sha256))
        {
            throw new InvalidDataException("The certified model contract is invalid.");
        }

        Uri download = new(Artifact.DownloadUrl, UriKind.Absolute);
        if (download.Scheme != Uri.UriSchemeHttps)
        {
            throw new InvalidDataException("The certified model URL must use HTTPS.");
        }
    }
}

public sealed class ModelArtifact
{
    public string Filename { get; init; } = string.Empty;

    public long Bytes { get; init; }

    public string Sha256 { get; init; } = string.Empty;

    public string DownloadUrl { get; init; } = string.Empty;

    public string DefaultRelativePath { get; init; } = string.Empty;
}
