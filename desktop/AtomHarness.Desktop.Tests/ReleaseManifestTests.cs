using System.IO.Compression;
using System.Security.Cryptography;
using System.Text;
using System.Text.Json;
using LucernaLabs.AtomHarness.Desktop;

namespace LucernaLabs.AtomHarness.Desktop.Tests;

[TestClass]
public sealed class ReleaseManifestTests
{
    [TestMethod]
    public async Task ManifestBindsEveryDeclaredFile()
    {
        string root = TestFiles.CreateRoot();
        try
        {
            await TestFiles.WriteReleaseManifestAsync(
                root,
                new Dictionary<string, byte[]>
                {
                    ["AtomHarness.Desktop.exe"] = Encoding.UTF8.GetBytes("shell"),
                    ["runtime/backend/backend.exe"] = Encoding.UTF8.GetBytes("backend"),
                });
            ReleaseManifest manifest = ReleaseManifest.Load(
                Path.Combine(root, "atom-harness-release-manifest.json"));
            await manifest.VerifyDirectoryAsync(root);
            File.AppendAllText(
                Path.Combine(root, "runtime", "backend", "backend.exe"),
                "tampered");
            await Assert.ThrowsExactlyAsync<InvalidDataException>(
                () => manifest.VerifyDirectoryAsync(root));
        }
        finally
        {
            Directory.Delete(root, true);
        }
    }

    [TestMethod]
    public async Task SafeUpdaterWaitsThenMovesVerifiedApplication()
    {
        string root = TestFiles.CreateRoot();
        try
        {
            string install = Path.Combine(root, "installed", "Atom Harness");
            Directory.CreateDirectory(install);
            File.WriteAllText(Path.Combine(install, "old.txt"), "old");
            string packageRoot = Path.Combine(root, "package");
            string appRoot = Path.Combine(packageRoot, "app");
            await TestFiles.WriteReleaseManifestAsync(
                appRoot,
                new Dictionary<string, byte[]>
                {
                    ["AtomHarness.Desktop.exe"] = Encoding.UTF8.GetBytes("new-shell"),
                    ["new.txt"] = Encoding.UTF8.GetBytes("new"),
                });
            string package = Path.Combine(root, "atom-harness.zip");
            ZipFile.CreateFromDirectory(packageRoot, package);
            string digest = await Integrity.Sha256FileAsync(package);
            string staging = Path.Combine(root, "updates");
            UpdateApplyResult result = await SafeUpdateInstaller.ApplyAsync(
                new UpdateApplyRequest(
                    package,
                    digest,
                    install,
                    staging,
                    0,
                    null));
            Assert.IsTrue(File.Exists(
                Path.Combine(install, "AtomHarness.Desktop.exe")));
            Assert.IsTrue(File.Exists(Path.Combine(install, "new.txt")));
            Assert.IsFalse(File.Exists(Path.Combine(install, "old.txt")));
            Assert.IsNotNull(result.BackupDirectory);
            Assert.IsTrue(File.Exists(
                Path.Combine(result.BackupDirectory, "old.txt")));
            Assert.IsTrue(File.Exists(result.ReceiptPath));
        }
        finally
        {
            Directory.Delete(root, true);
        }
    }

    [TestMethod]
    public async Task UnsafeStagingPathCannotReplaceInstall()
    {
        string root = TestFiles.CreateRoot();
        try
        {
            string install = Path.Combine(root, "installed");
            Directory.CreateDirectory(install);
            string package = Path.Combine(root, "update.zip");
            await File.WriteAllBytesAsync(package, [1, 2, 3]);
            string digest = Convert.ToHexStringLower(
                SHA256.HashData(await File.ReadAllBytesAsync(package)));
            await Assert.ThrowsExactlyAsync<InvalidDataException>(
                () => SafeUpdateInstaller.ApplyAsync(
                    new UpdateApplyRequest(
                        package,
                        digest,
                        install,
                        Path.Combine(install, "staging"),
                        0,
                        null)));
            Assert.IsTrue(Directory.Exists(install));
        }
        finally
        {
            Directory.Delete(root, true);
        }
    }

    [TestMethod]
    public void ManifestRejectsWindowsPathAliases()
    {
        string root = TestFiles.CreateRoot();
        try
        {
            string manifestPath = Path.Combine(
                root,
                "atom-harness-release-manifest.json");
            File.WriteAllText(
                manifestPath,
                $$"""
                {
                  "schema": 1,
                  "runtime": "atom-harness-release-manifest-v1",
                  "version": "5.1.0",
                  "files": [
                    {
                      "path": "runtime/CON.txt",
                      "bytes": 1,
                      "sha256": "{{new string('0', 64)}}"
                    }
                  ]
                }
                """);
            Assert.ThrowsExactly<InvalidDataException>(
                () => ReleaseManifest.Load(manifestPath));
        }
        finally
        {
            Directory.Delete(root, true);
        }
    }

    [TestMethod]
    public void RelativePathPolicyRejectsWindowsAliases()
    {
        Assert.IsTrue(Integrity.IsSafeRelativePath(
            "runtime/backend/atom-harness-backend.exe"));
        Assert.IsFalse(Integrity.IsSafeRelativePath("../escaped.txt"));
        Assert.IsFalse(Integrity.IsSafeRelativePath("runtime/file.txt:payload"));
        Assert.IsFalse(Integrity.IsSafeRelativePath("runtime/trailing."));
        Assert.IsFalse(Integrity.IsSafeRelativePath("runtime/trailing "));
        Assert.IsFalse(Integrity.IsSafeRelativePath("runtime/CON .txt"));
        Assert.IsFalse(Integrity.IsSafeRelativePath(@"C:\absolute.txt"));
    }

    [TestMethod]
    public async Task UpdateArchiveRejectsTraversalBeforeReplacement()
    {
        string root = TestFiles.CreateRoot();
        try
        {
            string install = Path.Combine(root, "installed", "Atom Harness");
            Directory.CreateDirectory(install);
            File.WriteAllText(Path.Combine(install, "old.txt"), "old");
            string package = Path.Combine(root, "unsafe-update.zip");
            using (ZipArchive archive = ZipFile.Open(
                package,
                ZipArchiveMode.Create))
            {
                ZipArchiveEntry entry = archive.CreateEntry(
                    "app/../../escaped.txt");
                await using StreamWriter writer = new(entry.Open());
                await writer.WriteAsync("unsafe");
            }

            string digest = await Integrity.Sha256FileAsync(package);
            await Assert.ThrowsExactlyAsync<InvalidDataException>(
                () => SafeUpdateInstaller.ApplyAsync(
                    new UpdateApplyRequest(
                        package,
                        digest,
                        install,
                        Path.Combine(root, "staging"),
                        0,
                        null)));
            Assert.IsTrue(File.Exists(Path.Combine(install, "old.txt")));
            Assert.IsFalse(File.Exists(Path.Combine(root, "escaped.txt")));
        }
        finally
        {
            Directory.Delete(root, true);
        }
    }
}

internal static class TestFiles
{
    internal static string CreateRoot()
    {
        string path = Path.Combine(
            Path.GetTempPath(),
            "atom-harness-desktop-tests",
            Guid.NewGuid().ToString("N"));
        Directory.CreateDirectory(path);
        return path;
    }

    internal static UpdatePolicyManifest LoadUpdateContract(
        string root,
        string version)
    {
        string path = Path.Combine(root, "lucerna-update.json");
        File.WriteAllText(path, UpdateContractJson(version));
        return UpdatePolicyManifest.Load(path);
    }

    internal static string UpdateContractJson(string version)
    {
        return $$"""
            {
              "schema": 1,
              "app_id": "com.lucernalabs.atom-harness",
              "app_name": "Atom Harness",
              "platform": "windows-x64",
              "current_version": "{{version}}",
              "feed_url": "https://updates.example/feed.json",
              "policy": {
                "automatic_download": false,
                "automatic_install": false,
                "explicit_user_consent_required": true,
                "artifact_sha256_required": true,
                "stage_outside_install_directory": true,
                "replace_only_after_app_exit": true,
                "rollback_backup_required": true
              },
              "runtime": "lucerna-release-client-v1"
            }
            """;
    }

    internal static async Task WriteReleaseManifestAsync(
        string root,
        IReadOnlyDictionary<string, byte[]> files)
    {
        Directory.CreateDirectory(root);
        List<object> manifestFiles = [];
        foreach ((string relativePath, byte[] content) in files)
        {
            string path = Path.Combine(
                root,
                relativePath.Replace('/', Path.DirectorySeparatorChar));
            Directory.CreateDirectory(
                Path.GetDirectoryName(path)
                ?? throw new InvalidDataException("Test path is invalid."));
            await File.WriteAllBytesAsync(path, content);
            manifestFiles.Add(new
            {
                path = relativePath,
                bytes = content.LongLength,
                sha256 = Convert.ToHexStringLower(SHA256.HashData(content)),
            });
        }

        await File.WriteAllTextAsync(
            Path.Combine(root, "atom-harness-release-manifest.json"),
            JsonSerializer.Serialize(
                new
                {
                    schema = 1,
                    runtime = "atom-harness-release-manifest-v1",
                    version = "5.1.0",
                    files = manifestFiles,
                },
                new JsonSerializerOptions
                {
                    WriteIndented = true,
                }));
    }
}
