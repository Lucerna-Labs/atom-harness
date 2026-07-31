using System.Text.Json;
using System.Text.RegularExpressions;

namespace LucernaLabs.AtomHarness.Desktop;

public sealed class UpdatePolicyManifest
{
    public int Schema { get; init; }

    public string AppId { get; init; } = string.Empty;

    public string AppName { get; init; } = string.Empty;

    public string Platform { get; init; } = string.Empty;

    public string CurrentVersion { get; init; } = string.Empty;

    public string FeedUrl { get; init; } = string.Empty;

    public UpdatePolicy Policy { get; init; } = new();

    public string Runtime { get; init; } = string.Empty;

    public static UpdatePolicyManifest Load(string path)
    {
        UpdatePolicyManifest manifest = JsonSerializer.Deserialize<UpdatePolicyManifest>(
            File.ReadAllText(path),
            JsonConfiguration.Options)
            ?? throw new InvalidDataException("The update contract is empty.");
        manifest.Validate();
        return manifest;
    }

    public void Validate()
    {
        if (Schema != 1
            || AppId != "com.lucernalabs.atom-harness"
            || AppName != "Atom Harness"
            || Platform != "windows-x64"
            || Runtime != "lucerna-release-client-v1")
        {
            throw new InvalidDataException("The update contract identity is invalid.");
        }

        _ = StableVersion.Parse(CurrentVersion);
        Uri feed = new(FeedUrl, UriKind.Absolute);
        if (feed.Scheme != Uri.UriSchemeHttps)
        {
            throw new InvalidDataException("The update feed must use HTTPS.");
        }

        if (!Policy.ExplicitUserConsentRequired
            || !Policy.ArtifactSha256Required
            || !Policy.StageOutsideInstallDirectory
            || !Policy.ReplaceOnlyAfterAppExit
            || !Policy.RollbackBackupRequired
            || Policy.AutomaticDownload
            || Policy.AutomaticInstall)
        {
            throw new InvalidDataException("The update safety policy is incomplete.");
        }
    }
}

public sealed class UpdatePolicy
{
    public bool AutomaticDownload { get; init; }

    public bool AutomaticInstall { get; init; }

    public bool ExplicitUserConsentRequired { get; init; }

    public bool ArtifactSha256Required { get; init; }

    public bool StageOutsideInstallDirectory { get; init; }

    public bool ReplaceOnlyAfterAppExit { get; init; }

    public bool RollbackBackupRequired { get; init; }
}

public sealed class UpdateFeed
{
    public int Schema { get; init; }

    public string AppId { get; init; } = string.Empty;

    public string Platform { get; init; } = string.Empty;

    public string Version { get; init; } = string.Empty;

    public string ReleaseNotes { get; init; } = string.Empty;

    public UpdateArtifact Artifact { get; init; } = new();
}

public sealed class UpdateArtifact
{
    public string Url { get; init; } = string.Empty;

    public long Bytes { get; init; }

    public string Sha256 { get; init; } = string.Empty;
}

public sealed record UpdateOffer(
    StableVersion Version,
    string ReleaseNotes,
    Uri ArtifactUrl,
    long ArtifactBytes,
    string ArtifactSha256);

public readonly record struct StableVersion(int Major, int Minor, int Patch)
    : IComparable<StableVersion>
{
    private static readonly Regex Pattern = new(
        @"^(0|[1-9]\d*)\.(0|[1-9]\d*)\.(0|[1-9]\d*)$",
        RegexOptions.CultureInvariant);

    public static StableVersion Parse(string value)
    {
        Match match = Pattern.Match(value);
        if (!match.Success)
        {
            throw new InvalidDataException("A stable three-part version is required.");
        }

        return new StableVersion(
            int.Parse(match.Groups[1].Value, System.Globalization.CultureInfo.InvariantCulture),
            int.Parse(match.Groups[2].Value, System.Globalization.CultureInfo.InvariantCulture),
            int.Parse(match.Groups[3].Value, System.Globalization.CultureInfo.InvariantCulture));
    }

    public int CompareTo(StableVersion other)
    {
        int major = Major.CompareTo(other.Major);
        if (major != 0)
        {
            return major;
        }

        int minor = Minor.CompareTo(other.Minor);
        return minor != 0 ? minor : Patch.CompareTo(other.Patch);
    }

    public override string ToString()
    {
        return $"{Major}.{Minor}.{Patch}";
    }
}

internal static class JsonConfiguration
{
    internal static readonly JsonSerializerOptions Options = new()
    {
        PropertyNamingPolicy = JsonNamingPolicy.SnakeCaseLower,
        PropertyNameCaseInsensitive = false,
        WriteIndented = true,
    };
}
