using System.Diagnostics;
using System.IO.Compression;
using System.Text.Json;

namespace LucernaLabs.AtomHarness.Desktop;

public sealed record UpdateApplyRequest(
    string PackagePath,
    string PackageSha256,
    string InstallDirectory,
    string StagingRoot,
    int WaitProcessId,
    string? RestartExecutable);

public sealed record UpdateApplyResult(
    string InstallDirectory,
    string? BackupDirectory,
    string ReceiptPath);

public static class SafeUpdateInstaller
{
    private const long MaximumExpandedBytes = 4L * 1024 * 1024 * 1024;
    private const int MaximumEntries = 10_000;

    public static async Task<UpdateApplyResult> ApplyAsync(
        UpdateApplyRequest request,
        CancellationToken cancellationToken = default)
    {
        string packagePath = Path.GetFullPath(request.PackagePath);
        string installDirectory = Path.GetFullPath(request.InstallDirectory)
            .TrimEnd(Path.DirectorySeparatorChar, Path.AltDirectorySeparatorChar);
        string stagingRoot = Path.GetFullPath(request.StagingRoot)
            .TrimEnd(Path.DirectorySeparatorChar, Path.AltDirectorySeparatorChar);
        ValidateTargets(packagePath, installDirectory, stagingRoot);
        if (!File.Exists(packagePath)
            || !Integrity.IsSha256(request.PackageSha256)
            || !string.Equals(
                await Integrity.Sha256FileAsync(packagePath, cancellationToken),
                request.PackageSha256,
                StringComparison.Ordinal))
        {
            throw new InvalidDataException("The update package SHA-256 is invalid.");
        }

        string operationRoot = Path.Combine(
            stagingRoot,
            "apply-" + Guid.NewGuid().ToString("N"));
        string extractedRoot = Path.Combine(operationRoot, "extracted");
        Directory.CreateDirectory(extractedRoot);
        try
        {
            await ExtractPackageAsync(
                packagePath,
                extractedRoot,
                cancellationToken);
            string stagedApplication = Path.Combine(extractedRoot, "app");
            string manifestPath = Path.Combine(
                stagedApplication,
                "atom-harness-release-manifest.json");
            if (!File.Exists(manifestPath))
            {
                throw new InvalidDataException(
                    "The update package has no release manifest.");
            }

            ReleaseManifest manifest = ReleaseManifest.Load(manifestPath);
            await manifest.VerifyDirectoryAsync(
                stagedApplication,
                cancellationToken);

            await WaitForExitAsync(request.WaitProcessId, cancellationToken);
            string? backupDirectory = null;
            bool oldInstallMoved = false;
            try
            {
                if (Directory.Exists(installDirectory))
                {
                    backupDirectory = installDirectory
                        + ".previous-"
                        + DateTime.UtcNow.ToString(
                            "yyyyMMddHHmmssfff",
                            System.Globalization.CultureInfo.InvariantCulture);
                    Directory.Move(installDirectory, backupDirectory);
                    oldInstallMoved = true;
                }

                Directory.Move(stagedApplication, installDirectory);
            }
            catch
            {
                if (!Directory.Exists(installDirectory)
                    && oldInstallMoved
                    && backupDirectory is not null
                    && Directory.Exists(backupDirectory))
                {
                    Directory.Move(backupDirectory, installDirectory);
                }

                throw;
            }

            string receiptPath = Path.Combine(
                stagingRoot,
                "last-update.json");
            string temporaryReceipt = receiptPath + ".tmp-" + Guid.NewGuid().ToString("N");
            string receipt = JsonSerializer.Serialize(
                new
                {
                    schema = 1,
                    runtime = "lucerna-safe-update-receipt-v1",
                    installed_at_utc = DateTime.UtcNow,
                    package_sha256 = request.PackageSha256,
                    install_directory = installDirectory,
                    backup_directory = backupDirectory,
                    manifest_version = manifest.Version,
                },
                JsonConfiguration.Options);
            await File.WriteAllTextAsync(
                temporaryReceipt,
                receipt,
                cancellationToken);
            File.Move(temporaryReceipt, receiptPath, true);

            if (!string.IsNullOrWhiteSpace(request.RestartExecutable))
            {
                string restartPath = Path.GetFullPath(
                    Path.Combine(installDirectory, request.RestartExecutable));
                if (!Integrity.IsWithin(restartPath, installDirectory)
                    || !File.Exists(restartPath))
                {
                    throw new InvalidDataException(
                        "The update restart target is invalid.");
                }

                Process.Start(new ProcessStartInfo(restartPath)
                {
                    UseShellExecute = true,
                    WorkingDirectory = installDirectory,
                });
            }

            return new UpdateApplyResult(
                installDirectory,
                backupDirectory,
                receiptPath);
        }
        finally
        {
            if (Directory.Exists(operationRoot))
            {
                Directory.Delete(operationRoot, true);
            }
        }
    }

    private static void ValidateTargets(
        string packagePath,
        string installDirectory,
        string stagingRoot)
    {
        string? installParent = Path.GetDirectoryName(installDirectory);
        if (string.IsNullOrWhiteSpace(installParent)
            || installDirectory == Path.GetPathRoot(installDirectory)
            || stagingRoot == Path.GetPathRoot(stagingRoot)
            || Integrity.IsWithin(stagingRoot, installDirectory)
            || Integrity.IsWithin(packagePath, installDirectory)
            || string.Equals(
                installDirectory,
                stagingRoot,
                StringComparison.OrdinalIgnoreCase))
        {
            throw new InvalidDataException("The update paths are unsafe.");
        }
    }

    private static async Task WaitForExitAsync(
        int processId,
        CancellationToken cancellationToken)
    {
        if (processId <= 0)
        {
            return;
        }

        try
        {
            using Process process = Process.GetProcessById(processId);
            await process.WaitForExitAsync(cancellationToken);
        }
        catch (ArgumentException)
        {
            // The application already exited, so replacement may proceed.
        }
    }

    private static async Task ExtractPackageAsync(
        string packagePath,
        string extractedRoot,
        CancellationToken cancellationToken)
    {
        using ZipArchive archive = ZipFile.OpenRead(packagePath);
        if (archive.Entries.Count == 0 || archive.Entries.Count > MaximumEntries)
        {
            throw new InvalidDataException("The update package entry count is invalid.");
        }

        long expandedBytes = 0;
        foreach (ZipArchiveEntry entry in archive.Entries)
        {
            cancellationToken.ThrowIfCancellationRequested();
            string name = entry.FullName.Replace('\\', '/');
            string pathForValidation = name.EndsWith("/", StringComparison.Ordinal)
                ? name[..^1]
                : name;
            if (!name.StartsWith("app/", StringComparison.Ordinal)
                || !Integrity.IsSafeRelativePath(pathForValidation))
            {
                throw new InvalidDataException(
                    "The update package contains an unsafe path.");
            }

            string targetPath = Path.GetFullPath(
                Path.Combine(
                    extractedRoot,
                    name.Replace('/', Path.DirectorySeparatorChar)));
            if (!Integrity.IsWithin(targetPath, extractedRoot))
            {
                throw new InvalidDataException(
                    "The update package escapes its staging directory.");
            }

            if (name.EndsWith("/", StringComparison.Ordinal))
            {
                Directory.CreateDirectory(targetPath);
                continue;
            }

            expandedBytes += entry.Length;
            if (expandedBytes > MaximumExpandedBytes)
            {
                throw new InvalidDataException(
                    "The expanded update package is oversized.");
            }

            Directory.CreateDirectory(
                Path.GetDirectoryName(targetPath)
                ?? throw new InvalidDataException("The update path is invalid."));
            await using Stream source = entry.Open();
            await using FileStream target = new(
                targetPath,
                FileMode.CreateNew,
                FileAccess.Write,
                FileShare.None,
                1024 * 1024,
                FileOptions.Asynchronous | FileOptions.WriteThrough);
            await source.CopyToAsync(target, cancellationToken);
            await target.FlushAsync(cancellationToken);
        }
    }
}
