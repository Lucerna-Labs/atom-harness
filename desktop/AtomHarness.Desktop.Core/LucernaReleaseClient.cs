using System.Net;
using System.Security.Cryptography;
using System.Text.Json;

namespace LucernaLabs.AtomHarness.Desktop;

public sealed class LucernaReleaseClient
{
    public const string Runtime = "lucerna-release-client-v1";

    private const int MaximumFeedBytes = 64 * 1024;
    private const long MaximumArtifactBytes = 2L * 1024 * 1024 * 1024;
    private readonly HttpClient _httpClient;
    private readonly UpdatePolicyManifest _manifest;
    private readonly string _stagingRoot;

    public LucernaReleaseClient(
        HttpClient httpClient,
        UpdatePolicyManifest manifest,
        string stagingRoot)
    {
        _httpClient = httpClient;
        _manifest = manifest;
        _manifest.Validate();
        _stagingRoot = Path.GetFullPath(stagingRoot);
    }

    public async Task<UpdateOffer?> CheckForUpdateAsync(
        CancellationToken cancellationToken = default)
    {
        Uri feedUri = new(_manifest.FeedUrl, UriKind.Absolute);
        using HttpResponseMessage response = await _httpClient.GetAsync(
            feedUri,
            HttpCompletionOption.ResponseHeadersRead,
            cancellationToken);
        response.EnsureSuccessStatusCode();
        RequireHttps(response.RequestMessage?.RequestUri);

        await using Stream stream = await response.Content.ReadAsStreamAsync(
            cancellationToken);
        byte[] payload = await ReadBoundedAsync(
            stream,
            MaximumFeedBytes,
            cancellationToken);
        UpdateFeed feed = JsonSerializer.Deserialize<UpdateFeed>(
            payload,
            JsonConfiguration.Options)
            ?? throw new InvalidDataException("The update feed is empty.");
        UpdateOffer offer = ValidateFeed(feed);
        StableVersion current = StableVersion.Parse(_manifest.CurrentVersion);
        return offer.Version.CompareTo(current) > 0 ? offer : null;
    }

    public async Task<string> DownloadAndVerifyAsync(
        UpdateOffer offer,
        IProgress<DownloadProgress>? progress = null,
        CancellationToken cancellationToken = default)
    {
        Directory.CreateDirectory(_stagingRoot);
        string finalPath = Path.Combine(
            _stagingRoot,
            $"atom-harness-{offer.Version}-windows-x64.zip");
        string partialPath = finalPath + ".download";
        if (File.Exists(finalPath)
            && new FileInfo(finalPath).Length == offer.ArtifactBytes
            && string.Equals(
                await Integrity.Sha256FileAsync(finalPath, cancellationToken),
                offer.ArtifactSha256,
                StringComparison.Ordinal))
        {
            return finalPath;
        }

        if (File.Exists(partialPath))
        {
            File.Delete(partialPath);
        }

        try
        {
            using HttpResponseMessage response = await _httpClient.GetAsync(
                offer.ArtifactUrl,
                HttpCompletionOption.ResponseHeadersRead,
                cancellationToken);
            response.EnsureSuccessStatusCode();
            RequireHttps(response.RequestMessage?.RequestUri);
            if (response.Content.Headers.ContentLength is long contentLength
                && contentLength != offer.ArtifactBytes)
            {
                throw new InvalidDataException(
                    "The update download length does not match its feed.");
            }

            await using Stream source = await response.Content.ReadAsStreamAsync(
                cancellationToken);
            await using FileStream target = new(
                partialPath,
                FileMode.CreateNew,
                FileAccess.Write,
                FileShare.None,
                1024 * 1024,
                FileOptions.Asynchronous
                    | FileOptions.SequentialScan
                    | FileOptions.WriteThrough);
            using IncrementalHash hasher = IncrementalHash.CreateHash(
                HashAlgorithmName.SHA256);
            byte[] buffer = new byte[1024 * 1024];
            long written = 0;
            while (true)
            {
                int read = await source.ReadAsync(buffer, cancellationToken);
                if (read == 0)
                {
                    break;
                }

                written += read;
                if (written > offer.ArtifactBytes)
                {
                    throw new InvalidDataException(
                        "The update download exceeded its declared length.");
                }

                await target.WriteAsync(buffer.AsMemory(0, read), cancellationToken);
                hasher.AppendData(buffer, 0, read);
                progress?.Report(new DownloadProgress(written, offer.ArtifactBytes));
            }

            await target.FlushAsync(cancellationToken);
            if (written != offer.ArtifactBytes)
            {
                throw new InvalidDataException(
                    "The update download is shorter than its declared length.");
            }

            string digest = Convert.ToHexStringLower(hasher.GetHashAndReset());
            if (!string.Equals(
                digest,
                offer.ArtifactSha256,
                StringComparison.Ordinal))
            {
                throw new CryptographicException(
                    "The update SHA-256 does not match its feed.");
            }

            await target.DisposeAsync();
            File.Move(partialPath, finalPath, true);
            return finalPath;
        }
        catch
        {
            if (File.Exists(partialPath))
            {
                File.Delete(partialPath);
            }

            throw;
        }
    }

    private UpdateOffer ValidateFeed(UpdateFeed feed)
    {
        if (feed.Schema != 1
            || feed.AppId != _manifest.AppId
            || feed.Platform != _manifest.Platform)
        {
            throw new InvalidDataException("The update feed identity is invalid.");
        }

        StableVersion version = StableVersion.Parse(feed.Version);
        if (feed.Artifact.Bytes <= 0 || feed.Artifact.Bytes > MaximumArtifactBytes)
        {
            throw new InvalidDataException("The update artifact size is invalid.");
        }

        if (!Integrity.IsSha256(feed.Artifact.Sha256))
        {
            throw new InvalidDataException("The update artifact SHA-256 is invalid.");
        }

        Uri artifactUri = new(feed.Artifact.Url, UriKind.Absolute);
        RequireHttps(artifactUri);
        return new UpdateOffer(
            version,
            feed.ReleaseNotes,
            artifactUri,
            feed.Artifact.Bytes,
            feed.Artifact.Sha256);
    }

    private static void RequireHttps(Uri? uri)
    {
        if (uri is null || uri.Scheme != Uri.UriSchemeHttps)
        {
            throw new InvalidDataException("Update transport must remain HTTPS.");
        }
    }

    private static async Task<byte[]> ReadBoundedAsync(
        Stream source,
        int maximumBytes,
        CancellationToken cancellationToken)
    {
        using MemoryStream target = new();
        byte[] buffer = new byte[8192];
        while (true)
        {
            int read = await source.ReadAsync(buffer, cancellationToken);
            if (read == 0)
            {
                return target.ToArray();
            }

            if (target.Length + read > maximumBytes)
            {
                throw new InvalidDataException("The update feed is oversized.");
            }

            target.Write(buffer, 0, read);
        }
    }
}

public readonly record struct DownloadProgress(long BytesReceived, long TotalBytes)
{
    public int Percent => TotalBytes <= 0
        ? 0
        : (int)Math.Clamp(BytesReceived * 100 / TotalBytes, 0, 100);
}
