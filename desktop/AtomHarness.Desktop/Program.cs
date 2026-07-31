namespace LucernaLabs.AtomHarness.Desktop;

internal static class Program
{
    internal const string DesktopRuntime = "atom-harness-desktop-v5";
    internal const string AuthorityRuntime = "ATOM_HARNESS_OPERATOR_RUNTIME";
    internal const string WikiRuntime = "ATOM_HARNESS_WIKI_RUNTIME";
    internal const string RagRuntime = "ATOM_HARNESS_RAG_RUNTIME";
    internal const string SideViewRuntime = "ATOM_HARNESS_OPERATOR_UI_RUNTIME";
    internal const string ArtifactBindingMarker = "render_operator_surface";

    [STAThread]
    private static int Main(string[] arguments)
    {
        DesktopPaths paths = DesktopPaths.Create();
        if (arguments.Length == 2
            && arguments[0] == "--verify-install")
        {
            return InstalledLayoutVerifier.VerifyAsync(paths, arguments[1])
                .GetAwaiter()
                .GetResult()
                ? 0
                : 1;
        }

        using Mutex instance = new(
            true,
            @"Local\LucernaLabs.AtomHarness.Desktop.v5",
            out bool createdNew);
        if (!createdNew)
        {
            MessageBox.Show(
                "Atom Harness is already running.",
                "Atom Harness",
                MessageBoxButtons.OK,
                MessageBoxIcon.Information);
            return 0;
        }

        ApplicationConfiguration.Initialize();
        Application.SetUnhandledExceptionMode(
            UnhandledExceptionMode.CatchException);
        SafeDiagnostics diagnostics = new(
            Path.Combine(paths.LogsRoot, "desktop-events.jsonl"));
        Application.ThreadException += (_, eventArguments) =>
        {
            string identity = diagnostics.ErrorIdentity(eventArguments.Exception);
            _ = diagnostics.RecordAsync(
                "desktop-unhandled",
                eventArguments.Exception.ToString());
            MessageBox.Show(
                $"Atom Harness encountered an unexpected error ({identity[..12]}).",
                "Atom Harness",
                MessageBoxButtons.OK,
                MessageBoxIcon.Error);
        };
        AppDomain.CurrentDomain.UnhandledException += (_, eventArguments) =>
        {
            if (eventArguments.ExceptionObject is Exception error)
            {
                _ = diagnostics.RecordAsync("desktop-fatal", error.ToString());
            }
        };

        Application.Run(new MainForm(paths, diagnostics));
        GC.KeepAlive(instance);
        return 0;
    }
}
