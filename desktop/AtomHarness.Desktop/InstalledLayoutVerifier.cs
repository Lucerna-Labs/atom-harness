using System.Security.Cryptography;
using System.Text;
using System.Text.Json;
using Microsoft.Web.WebView2.Core;

namespace LucernaLabs.AtomHarness.Desktop;

internal static class InstalledLayoutVerifier
{
    internal static async Task<bool> VerifyAsync(
        DesktopPaths paths,
        string reportPath,
        CancellationToken cancellationToken = default)
    {
        object report;
        bool passed;
        try
        {
            foreach (string required in new[]
            {
                paths.BackendExecutable,
                paths.LlamaServerExecutable,
                paths.UpdaterExecutable,
                paths.ModelContractPath,
                paths.UpdateContractPath,
                paths.ReleaseManifestPath,
            })
            {
                if (!File.Exists(required))
                {
                    throw new FileNotFoundException(
                        "An installed Atom Harness file is absent.",
                        required);
                }
            }

            UpdatePolicyManifest update = UpdatePolicyManifest.Load(
                paths.UpdateContractPath);
            CertifiedModelContract model = CertifiedModelContract.Load(
                paths.ModelContractPath);
            ReleaseManifest release = ReleaseManifest.Load(
                paths.ReleaseManifestPath);
            await release.VerifyDirectoryAsync(
                paths.InstallRoot,
                cancellationToken);
            string? webViewVersion =
                CoreWebView2Environment.GetAvailableBrowserVersionString();
            if (string.IsNullOrWhiteSpace(webViewVersion))
            {
                throw new InvalidOperationException(
                    "Microsoft Edge WebView2 Runtime is unavailable.");
            }

            report = new
            {
                schema = 1,
                runtime = "atom-harness-desktop-install-verification-v1",
                passed = true,
                desktop_runtime = Program.DesktopRuntime,
                authority_runtime = Program.AuthorityRuntime,
                update_runtime = update.Runtime,
                release_version = release.Version,
                model_sha256 = model.Artifact.Sha256,
                webview2_version = webViewVersion,
                install_root = paths.InstallRoot,
                verified_at_utc = DateTime.UtcNow,
            };
            passed = true;
        }
        catch (Exception error)
        {
            report = new
            {
                schema = 1,
                runtime = "atom-harness-desktop-install-verification-v1",
                passed = false,
                error_sha256 = ErrorIdentity(error),
                verified_at_utc = DateTime.UtcNow,
            };
            passed = false;
        }

        string fullReportPath = Path.GetFullPath(reportPath);
        Directory.CreateDirectory(
            Path.GetDirectoryName(fullReportPath)
            ?? throw new InvalidDataException("The verification report path is invalid."));
        string temporary = fullReportPath + ".tmp-" + Guid.NewGuid().ToString("N");
        await File.WriteAllTextAsync(
            temporary,
            JsonSerializer.Serialize(
                report,
                new JsonSerializerOptions
                {
                    PropertyNamingPolicy = JsonNamingPolicy.SnakeCaseLower,
                    WriteIndented = true,
                }),
            cancellationToken);
        File.Move(temporary, fullReportPath, true);
        return passed;
    }

    private static string ErrorIdentity(Exception error)
    {
        byte[] digest = SHA256.HashData(
            Encoding.UTF8.GetBytes(error.GetType().FullName + "\n" + error.Message));
        return Convert.ToHexStringLower(digest);
    }
}
