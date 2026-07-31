namespace LucernaLabs.AtomHarness.Desktop;

internal sealed record DesktopPaths(
    string InstallRoot,
    string StateRoot,
    string SessionRoot,
    string LogsRoot,
    string UpdatesRoot,
    string ModelRoot,
    string WebViewRoot,
    string SettingsPath,
    string BackendExecutable,
    string LlamaServerExecutable,
    string UpdaterExecutable,
    string ModelContractPath,
    string UpdateContractPath,
    string ReleaseManifestPath)
{
    internal static DesktopPaths Create()
    {
        string installRoot = Path.GetFullPath(AppContext.BaseDirectory)
            .TrimEnd(Path.DirectorySeparatorChar, Path.AltDirectorySeparatorChar);
        string local = Environment.GetFolderPath(
            Environment.SpecialFolder.LocalApplicationData);
        string stateRoot = Path.Combine(local, "Lucerna Labs", "Atom Harness");
        return new DesktopPaths(
            installRoot,
            stateRoot,
            Path.Combine(stateRoot, "Data", "Sessions", "default"),
            Path.Combine(stateRoot, "Data", "Logs"),
            Path.Combine(stateRoot, "Data", "Updates"),
            Path.Combine(stateRoot, "Models", "Qwen3-4B-Instruct-2507-Q8_0"),
            Path.Combine(stateRoot, "WebView2"),
            Path.Combine(stateRoot, "settings.json"),
            Path.Combine(
                installRoot,
                "runtime",
                "backend",
                "atom-harness-backend.exe"),
            Path.Combine(installRoot, "runtime", "llama", "llama-server.exe"),
            Path.Combine(installRoot, "tools", "AtomHarness.Updater.exe"),
            Path.Combine(installRoot, "atom-language-model.json"),
            Path.Combine(installRoot, "lucerna-update.json"),
            Path.Combine(installRoot, "atom-harness-release-manifest.json"));
    }

    internal IEnumerable<string> ModelCandidates(
        CertifiedModelContract contract,
        DesktopSettings settings)
    {
        if (!string.IsNullOrWhiteSpace(settings.ModelPath))
        {
            yield return settings.ModelPath;
        }

        string? environmentPath = Environment.GetEnvironmentVariable(
            "ATOM_LLM_MODEL_PATH");
        if (!string.IsNullOrWhiteSpace(environmentPath))
        {
            yield return environmentPath;
        }

        yield return Path.Combine(ModelRoot, contract.Artifact.Filename);
        yield return Path.Combine(
            @"C:\Projects\atom-harness-models\Qwen3-4B-Instruct-2507-Q8_0",
            contract.Artifact.Filename);
        yield return Path.GetFullPath(
            Path.Combine(InstallRoot, contract.Artifact.DefaultRelativePath));
    }
}
