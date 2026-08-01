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
            string knowledgeManifestPath = ResolveKnowledgeManifest(paths.InstallRoot);
            foreach (string required in new[]
            {
                paths.BackendExecutable,
                paths.LlamaServerExecutable,
                paths.UpdaterExecutable,
                paths.ModelContractPath,
                paths.UpdateContractPath,
                paths.ReleaseManifestPath,
                knowledgeManifestPath,
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
            KnowledgePackVerification knowledge = VerifyKnowledgePack(
                knowledgeManifestPath);
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
                knowledge_pack_id = knowledge.PackId,
                knowledge_domain_count = knowledge.DomainCount,
                knowledge_claim_count = knowledge.ClaimCount,
                knowledge_source_count = knowledge.SourceCount,
                knowledge_manifest_sha256 = knowledge.ManifestSha256,
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

    private static string ResolveKnowledgeManifest(string installRoot)
    {
        string[] candidates =
        {
            Path.Combine(
                installRoot,
                "runtime",
                "backend",
                "_internal",
                "knowledge_packs",
                "universal-foundation-v1",
                "manifest.json"),
            Path.Combine(
                installRoot,
                "runtime",
                "backend",
                "knowledge_packs",
                "universal-foundation-v1",
                "manifest.json"),
        };
        return candidates.FirstOrDefault(File.Exists) ?? candidates[0];
    }

    private static KnowledgePackVerification VerifyKnowledgePack(
        string manifestPath)
    {
        string fullManifest = Path.GetFullPath(manifestPath);
        if ((File.GetAttributes(fullManifest) & FileAttributes.ReparsePoint) != 0)
        {
            throw new InvalidDataException(
                "The installed knowledge manifest cannot be a reparse point.");
        }
        string packRoot = Path.GetDirectoryName(fullManifest)
            ?? throw new InvalidDataException("Knowledge pack root is invalid.");
        using JsonDocument document = JsonDocument.Parse(
            File.ReadAllBytes(fullManifest));
        JsonElement root = document.RootElement;
        if (root.GetProperty("schema").GetInt32() != 1)
        {
            throw new InvalidDataException("Knowledge pack schema is invalid.");
        }
        string packId = root.GetProperty("pack_id").GetString()
            ?? throw new InvalidDataException("Knowledge pack identity is absent.");
        JsonElement hashes = root.GetProperty("file_sha256");
        foreach (JsonProperty entry in hashes.EnumerateObject())
        {
            string candidate = Path.GetFullPath(
                Path.Combine(packRoot, entry.Name.Replace('/', Path.DirectorySeparatorChar)));
            if (!candidate.StartsWith(
                packRoot + Path.DirectorySeparatorChar,
                StringComparison.OrdinalIgnoreCase))
            {
                throw new InvalidDataException("Knowledge pack file escapes its root.");
            }
            if (!File.Exists(candidate)
                || (File.GetAttributes(candidate) & FileAttributes.ReparsePoint) != 0)
            {
                throw new FileNotFoundException(
                    "An installed knowledge file is absent or unsafe.",
                    candidate);
            }
            string expected = entry.Value.GetString() ?? string.Empty;
            string actual = Convert.ToHexStringLower(
                SHA256.HashData(File.ReadAllBytes(candidate)));
            if (!CryptographicOperations.FixedTimeEquals(
                Encoding.ASCII.GetBytes(expected),
                Encoding.ASCII.GetBytes(actual)))
            {
                throw new InvalidDataException(
                    "An installed knowledge file failed SHA-256 verification.");
            }
        }

        string taxonomyPath = Path.Combine(
            packRoot,
            root.GetProperty("taxonomy_file").GetString()
                ?? throw new InvalidDataException("Taxonomy file is absent."));
        string sourcesPath = Path.Combine(
            packRoot,
            root.GetProperty("sources_file").GetString()
                ?? throw new InvalidDataException("Sources file is absent."));
        using JsonDocument taxonomy = JsonDocument.Parse(
            File.ReadAllBytes(taxonomyPath));
        using JsonDocument sources = JsonDocument.Parse(
            File.ReadAllBytes(sourcesPath));
        int domainCount = taxonomy.RootElement.GetProperty("domains").GetArrayLength();
        int sourceCount = sources.RootElement.GetProperty("sources").GetArrayLength();
        int claimCount = 0;
        foreach (JsonElement shard in root.GetProperty("claim_shards").EnumerateArray())
        {
            string relative = shard.GetString()
                ?? throw new InvalidDataException("Knowledge shard path is invalid.");
            string shardPath = Path.Combine(
                packRoot,
                relative.Replace('/', Path.DirectorySeparatorChar));
            claimCount += File.ReadLines(shardPath).Count(
                line => !string.IsNullOrWhiteSpace(line));
        }
        if (domainCount != 15 || claimCount < 45 || sourceCount < 20)
        {
            throw new InvalidDataException(
                "The installed knowledge pack is below the Phase 7 coverage floor.");
        }
        return new KnowledgePackVerification(
            packId,
            domainCount,
            claimCount,
            sourceCount,
            Convert.ToHexStringLower(
                SHA256.HashData(File.ReadAllBytes(fullManifest))));
    }

    private sealed record KnowledgePackVerification(
        string PackId,
        int DomainCount,
        int ClaimCount,
        int SourceCount,
        string ManifestSha256);
}
