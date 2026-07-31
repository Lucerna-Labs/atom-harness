using System.Net;
using System.Security.Cryptography;

namespace LucernaLabs.AtomHarness.Desktop;

internal sealed class ModelProvisioner(HttpClient httpClient)
{
    internal async Task<string?> FindVerifiedModelAsync(
        IEnumerable<string> candidates,
        CertifiedModelContract contract,
        IProgress<ModelProgress>? progress = null,
        CancellationToken cancellationToken = default)
    {
        HashSet<string> seen = new(StringComparer.OrdinalIgnoreCase);
        foreach (string candidate in candidates)
        {
            string path;
            try
            {
                path = Path.GetFullPath(candidate);
            }
            catch (Exception error) when (
                error is ArgumentException
                    or NotSupportedException
                    or PathTooLongException)
            {
                continue;
            }

            if (!seen.Add(path) || !File.Exists(path))
            {
                continue;
            }

            FileInfo info = new(path);
            if (info.Length != contract.Artifact.Bytes)
            {
                continue;
            }

            progress?.Report(new ModelProgress(
                "Verifying the certified language model",
                0));
            string digest = await Integrity.Sha256FileAsync(path, cancellationToken);
            if (string.Equals(
                digest,
                contract.Artifact.Sha256,
                StringComparison.Ordinal))
            {
                progress?.Report(new ModelProgress("Model verified", 100));
                return path;
            }
        }

        return null;
    }

    internal async Task<string> DownloadVerifiedModelAsync(
        CertifiedModelContract contract,
        string modelRoot,
        string stagingRoot,
        string installRoot,
        IProgress<ModelProgress>? progress = null,
        CancellationToken cancellationToken = default)
    {
        string fullModelRoot = Path.GetFullPath(modelRoot);
        string fullStagingRoot = Path.GetFullPath(stagingRoot);
        if (Integrity.IsWithin(fullStagingRoot, installRoot)
            || Integrity.IsWithin(fullModelRoot, installRoot))
        {
            throw new InvalidDataException(
                "Model download storage must remain outside the install directory.");
        }

        Directory.CreateDirectory(fullModelRoot);
        Directory.CreateDirectory(fullStagingRoot);
        string finalPath = Path.Combine(
            fullModelRoot,
            contract.Artifact.Filename);
        string partialPath = Path.Combine(
            fullStagingRoot,
            contract.Artifact.Filename + "." + Guid.NewGuid().ToString("N") + ".part");
        try
        {
            Uri uri = new(contract.Artifact.DownloadUrl, UriKind.Absolute);
            if (uri.Scheme != Uri.UriSchemeHttps)
            {
                throw new InvalidDataException("Model download transport must be HTTPS.");
            }

            using HttpResponseMessage response = await httpClient.GetAsync(
                uri,
                HttpCompletionOption.ResponseHeadersRead,
                cancellationToken);
            response.EnsureSuccessStatusCode();
            if (response.RequestMessage?.RequestUri?.Scheme != Uri.UriSchemeHttps)
            {
                throw new InvalidDataException(
                    "Model download redirected away from HTTPS.");
            }

            if (response.Content.Headers.ContentLength is long length
                && length != contract.Artifact.Bytes)
            {
                throw new InvalidDataException(
                    "The model download length does not match its contract.");
            }

            await using Stream source = await response.Content.ReadAsStreamAsync(
                cancellationToken);
            await using FileStream target = new(
                partialPath,
                FileMode.CreateNew,
                FileAccess.Write,
                FileShare.None,
                4 * 1024 * 1024,
                FileOptions.Asynchronous
                    | FileOptions.SequentialScan
                    | FileOptions.WriteThrough);
            using IncrementalHash hasher = IncrementalHash.CreateHash(
                HashAlgorithmName.SHA256);
            byte[] buffer = new byte[4 * 1024 * 1024];
            long written = 0;
            while (true)
            {
                int read = await source.ReadAsync(buffer, cancellationToken);
                if (read == 0)
                {
                    break;
                }

                written += read;
                if (written > contract.Artifact.Bytes)
                {
                    throw new InvalidDataException(
                        "The model download exceeded its declared size.");
                }

                await target.WriteAsync(buffer.AsMemory(0, read), cancellationToken);
                hasher.AppendData(buffer, 0, read);
                int percent = (int)Math.Clamp(
                    written * 100 / contract.Artifact.Bytes,
                    0,
                    100);
                progress?.Report(new ModelProgress(
                    $"Downloading the certified model ({percent}%)",
                    percent));
            }

            await target.FlushAsync(cancellationToken);
            if (written != contract.Artifact.Bytes)
            {
                throw new InvalidDataException(
                    "The model download is incomplete.");
            }

            string digest = Convert.ToHexStringLower(hasher.GetHashAndReset());
            if (!string.Equals(
                digest,
                contract.Artifact.Sha256,
                StringComparison.Ordinal))
            {
                throw new CryptographicException(
                    "The model download failed SHA-256 verification.");
            }

            await target.DisposeAsync();
            File.Move(partialPath, finalPath, true);
            progress?.Report(new ModelProgress("Model downloaded and verified", 100));
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
}

internal readonly record struct ModelProgress(string Message, int Percent);
