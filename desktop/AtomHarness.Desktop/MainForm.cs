using System.Diagnostics;
using System.Net;
using System.Security.Cryptography;
using Microsoft.Web.WebView2.Core;
using Microsoft.Web.WebView2.WinForms;

namespace LucernaLabs.AtomHarness.Desktop;

internal sealed class MainForm : Form
{
    private readonly DesktopPaths _paths;
    private readonly SafeDiagnostics _diagnostics;
    private readonly ToolStripStatusLabel _status = new()
    {
        Text = "Starting",
        Spring = true,
        TextAlign = ContentAlignment.MiddleLeft,
    };
    private readonly ToolStripButton _checkUpdates = new("Check for updates");
    private readonly ToolStripButton _openData = new("Open data");
    private readonly WebView2 _webView = new()
    {
        Dock = DockStyle.Fill,
        Visible = false,
    };
    private readonly Panel _startupPanel = new()
    {
        Dock = DockStyle.Fill,
        BackColor = Color.FromArgb(10, 16, 28),
    };
    private readonly Label _startupTitle = new()
    {
        AutoSize = false,
        Dock = DockStyle.Top,
        Height = 90,
        Text = "ATOM HARNESS",
        TextAlign = ContentAlignment.BottomCenter,
        Font = new Font("Segoe UI Semibold", 24, FontStyle.Bold),
        ForeColor = Color.FromArgb(238, 245, 255),
    };
    private readonly Label _startupMessage = new()
    {
        AutoSize = false,
        Dock = DockStyle.Top,
        Height = 70,
        Text = "Preparing the certified local runtime",
        TextAlign = ContentAlignment.MiddleCenter,
        Font = new Font("Segoe UI", 11),
        ForeColor = Color.FromArgb(160, 180, 205),
    };
    private readonly ProgressBar _startupProgress = new()
    {
        Width = 360,
        Height = 10,
        Style = ProgressBarStyle.Marquee,
        MarqueeAnimationSpeed = 28,
    };
    private BackendSupervisor? _backend;
    private Uri? _origin;
    private bool _closing;
    private bool _allowClose;
    private bool _updatePrepared;

    internal MainForm(
        DesktopPaths paths,
        SafeDiagnostics diagnostics)
    {
        _paths = paths;
        _diagnostics = diagnostics;
        Text = "Atom Harness";
        Width = 1420;
        Height = 900;
        MinimumSize = new Size(1040, 680);
        StartPosition = FormStartPosition.CenterScreen;
        BackColor = Color.FromArgb(10, 16, 28);

        StatusStrip statusStrip = new();
        statusStrip.Items.Add(_status);
        ToolStrip toolbar = new()
        {
            GripStyle = ToolStripGripStyle.Hidden,
            BackColor = Color.FromArgb(17, 27, 44),
            ForeColor = Color.White,
            Renderer = new ToolStripProfessionalRenderer(
                new DarkColorTable()),
        };
        toolbar.Items.Add(new ToolStripLabel("Atom Harness Desktop 6")
        {
            Font = new Font("Segoe UI Semibold", 10, FontStyle.Bold),
        });
        toolbar.Items.Add(new ToolStripSeparator());
        toolbar.Items.Add(_checkUpdates);
        toolbar.Items.Add(_openData);
        _checkUpdates.Enabled = false;
        _checkUpdates.Click += CheckUpdates;
        _openData.Click += OpenData;

        FlowLayoutPanel progressHost = new()
        {
            Dock = DockStyle.Top,
            Height = 60,
            FlowDirection = FlowDirection.LeftToRight,
            WrapContents = false,
            Padding = new Padding(0, 22, 0, 0),
        };
        progressHost.Controls.Add(_startupProgress);
        progressHost.Resize += (_, _) =>
        {
            _startupProgress.Margin = new Padding(
                Math.Max(0, (progressHost.ClientSize.Width - _startupProgress.Width) / 2),
                0,
                0,
                0);
        };
        _startupPanel.Controls.Add(progressHost);
        _startupPanel.Controls.Add(_startupMessage);
        _startupPanel.Controls.Add(_startupTitle);

        Controls.Add(_webView);
        Controls.Add(_startupPanel);
        Controls.Add(toolbar);
        Controls.Add(statusStrip);
        toolbar.Dock = DockStyle.Top;
        statusStrip.Dock = DockStyle.Bottom;
        Shown += StartRuntime;
        FormClosing += BeginClosing;
    }

    private async void StartRuntime(object? sender, EventArgs eventArguments)
    {
        try
        {
            UpdateStartup("Checking the installed runtime", null);
            RequireInstalledFiles();
            CertifiedModelContract modelContract = CertifiedModelContract.Load(
                _paths.ModelContractPath);
            DesktopSettings settings = DesktopSettings.Load(_paths.SettingsPath);
            using HttpClient httpClient = CreateHttpClient();
            ModelProvisioner provisioner = new(httpClient);
            Progress<ModelProgress> modelProgress = new(progress =>
                UpdateStartup(progress.Message, progress.Percent));
            string? modelPath = await provisioner.FindVerifiedModelAsync(
                _paths.ModelCandidates(modelContract, settings),
                modelContract,
                modelProgress);
            if (modelPath is null)
            {
                modelPath = await ResolveMissingModelAsync(
                    provisioner,
                    modelContract,
                    modelProgress);
            }

            settings.ModelPath = modelPath;
            await settings.SaveAsync(_paths.SettingsPath);
            UpdateStartup(
                "Preloading Atom, Qwen, and the permission registry",
                null);
            await InitializeWebViewAsync();
            _backend = new BackendSupervisor(_paths, _diagnostics);
            BackendStartup startup = await _backend.StartAsync(
                modelPath,
                settings.GpuLayers);
            _origin = startup.Origin;
            _webView.CoreWebView2.Navigate(_origin.AbsoluteUri);
            _webView.Visible = true;
            _startupPanel.Visible = false;
            _status.Text = "Ready, local model resident";
            _checkUpdates.Enabled = true;
        }
        catch (OperationCanceledException)
        {
            _allowClose = true;
            Close();
        }
        catch (Exception error)
        {
            await _diagnostics.RecordAsync("desktop-startup", error.ToString());
            string identity = _diagnostics.ErrorIdentity(error);
            UpdateStartup(
                $"Startup could not finish. Error reference {identity[..12]}.",
                0);
            _status.Text = "Startup failed";
            _startupProgress.Style = ProgressBarStyle.Blocks;
            _startupProgress.Value = 0;
        }
    }

    private async Task<string> ResolveMissingModelAsync(
        ModelProvisioner provisioner,
        CertifiedModelContract contract,
        IProgress<ModelProgress> progress)
    {
        DialogResult choice = MessageBox.Show(
            "The certified 4.28 GB language model was not found.\n\n"
                + "Choose Yes to download and verify it, No to locate an existing "
                + "copy, or Cancel to close Atom Harness.",
            "Set up the Atom language model",
            MessageBoxButtons.YesNoCancel,
            MessageBoxIcon.Information);
        if (choice == DialogResult.Cancel)
        {
            throw new OperationCanceledException();
        }

        if (choice == DialogResult.No)
        {
            using OpenFileDialog dialog = new()
            {
                Title = "Locate the certified Qwen GGUF model",
                Filter = "GGUF model (*.gguf)|*.gguf",
                CheckFileExists = true,
                Multiselect = false,
            };
            if (dialog.ShowDialog(this) != DialogResult.OK)
            {
                throw new OperationCanceledException();
            }

            string? selected = await provisioner.FindVerifiedModelAsync(
                [dialog.FileName],
                contract,
                progress);
            if (selected is null)
            {
                throw new InvalidDataException(
                    "The selected file is not the certified Atom language model.");
            }

            return selected;
        }

        string modelStaging = Path.Combine(
            _paths.StateRoot,
            "Data",
            "ModelStaging");
        return await provisioner.DownloadVerifiedModelAsync(
            contract,
            _paths.ModelRoot,
            modelStaging,
            _paths.InstallRoot,
            progress);
    }

    private async Task InitializeWebViewAsync()
    {
        Directory.CreateDirectory(_paths.WebViewRoot);
        CoreWebView2Environment environment =
            await CoreWebView2Environment.CreateAsync(
                null,
                _paths.WebViewRoot);
        await _webView.EnsureCoreWebView2Async(environment);
        _webView.CoreWebView2.Settings.AreDevToolsEnabled = false;
        _webView.CoreWebView2.Settings.IsStatusBarEnabled = false;
        _webView.CoreWebView2.Settings.AreDefaultContextMenusEnabled = false;
        _webView.CoreWebView2.Settings.IsPasswordAutosaveEnabled = false;
        _webView.CoreWebView2.Settings.IsGeneralAutofillEnabled = false;
        _webView.CoreWebView2.NewWindowRequested += (_, arguments) =>
        {
            arguments.Handled = true;
        };
        _webView.CoreWebView2.NavigationStarting += (_, arguments) =>
        {
            if (_origin is null)
            {
                return;
            }

            if (!Uri.TryCreate(arguments.Uri, UriKind.Absolute, out Uri? target)
                || target.Scheme != Uri.UriSchemeHttp
                || target.Host != _origin.Host
                || target.Port != _origin.Port)
            {
                arguments.Cancel = true;
            }
        };
    }

    private void RequireInstalledFiles()
    {
        foreach (string required in new[]
        {
            _paths.BackendExecutable,
            _paths.LlamaServerExecutable,
            _paths.UpdaterExecutable,
            _paths.ModelContractPath,
            _paths.UpdateContractPath,
        })
        {
            if (!File.Exists(required))
            {
                throw new FileNotFoundException(
                    "The Atom Harness installation is incomplete.",
                    required);
            }
        }

        string? webViewVersion =
            CoreWebView2Environment.GetAvailableBrowserVersionString();
        if (string.IsNullOrWhiteSpace(webViewVersion))
        {
            throw new InvalidOperationException(
                "Microsoft Edge WebView2 Runtime is unavailable.");
        }
    }

    private async void CheckUpdates(object? sender, EventArgs eventArguments)
    {
        if (_closing)
        {
            return;
        }

        _checkUpdates.Enabled = false;
        _status.Text = "Checking for updates";
        try
        {
            UpdatePolicyManifest manifest = UpdatePolicyManifest.Load(
                _paths.UpdateContractPath);
            using HttpClient httpClient = CreateHttpClient();
            LucernaReleaseClient client = new(
                httpClient,
                manifest,
                Path.Combine(_paths.UpdatesRoot, "Downloads"));
            UpdateOffer? offer = await client.CheckForUpdateAsync();
            if (offer is null)
            {
                MessageBox.Show(
                    "Atom Harness is up to date.",
                    "Updates",
                    MessageBoxButtons.OK,
                    MessageBoxIcon.Information);
                return;
            }

            DialogResult confirmUpdate = MessageBox.Show(
                $"Atom Harness {offer.Version} is available.\n\n"
                    + offer.ReleaseNotes
                    + "\n\nDownload and install this update after Atom Harness closes?",
                "Install update",
                MessageBoxButtons.YesNo,
                MessageBoxIcon.Question);
            if (confirmUpdate != DialogResult.Yes)
            {
                return;
            }

            Progress<DownloadProgress> progress = new(value =>
                _status.Text = $"Downloading update, {value.Percent}%");
            string package = await client.DownloadAndVerifyAsync(
                offer,
                progress);
            DialogResult confirmInstall = MessageBox.Show(
                $"Atom Harness {offer.Version} has been downloaded and verified.\n\n"
                    + "Install it now? Atom Harness will close, the updater will "
                    + "retain the current installation for rollback, and the new "
                    + "version will restart after replacement succeeds.",
                "Confirm verified update",
                MessageBoxButtons.YesNo,
                MessageBoxIcon.Question);
            if (confirmInstall != DialogResult.Yes)
            {
                _status.Text = "Verified update retained, installation canceled";
                return;
            }

            await PrepareUpdateAsync(package, offer);
        }
        catch (Exception error)
        {
            await _diagnostics.RecordAsync("desktop-update", error.ToString());
            string identity = _diagnostics.ErrorIdentity(error);
            MessageBox.Show(
                $"The update check could not finish ({identity[..12]}). "
                    + "The installed application was not changed.",
                "Updates",
                MessageBoxButtons.OK,
                MessageBoxIcon.Warning);
        }
        finally
        {
            if (!_updatePrepared)
            {
                _status.Text = "Ready, local model resident";
                _checkUpdates.Enabled = true;
            }
        }
    }

    private async Task PrepareUpdateAsync(
        string packagePath,
        UpdateOffer offer)
    {
        string helperRoot = Path.Combine(
            _paths.UpdatesRoot,
            "Helpers",
            Guid.NewGuid().ToString("N"));
        Directory.CreateDirectory(helperRoot);
        string helperPath = Path.Combine(helperRoot, "AtomHarness.Updater.exe");
        File.Copy(_paths.UpdaterExecutable, helperPath, false);
        string sourceHash = await Integrity.Sha256FileAsync(
            _paths.UpdaterExecutable);
        string helperHash = await Integrity.Sha256FileAsync(helperPath);
        if (!string.Equals(sourceHash, helperHash, StringComparison.Ordinal))
        {
            throw new CryptographicException(
                "The staged updater failed SHA-256 verification.");
        }

        ProcessStartInfo start = new(helperPath)
        {
            UseShellExecute = false,
            CreateNoWindow = true,
            WorkingDirectory = helperRoot,
        };
        foreach (string argument in new[]
        {
            "--package",
            packagePath,
            "--sha256",
            offer.ArtifactSha256,
            "--install-dir",
            _paths.InstallRoot,
            "--staging-root",
            _paths.UpdatesRoot,
            "--wait-pid",
            Environment.ProcessId.ToString(
                System.Globalization.CultureInfo.InvariantCulture),
            "--restart",
            "AtomHarness.Desktop.exe",
        })
        {
            start.ArgumentList.Add(argument);
        }

        _ = Process.Start(start)
            ?? throw new InvalidOperationException("The update helper did not start.");
        _updatePrepared = true;
        _status.Text = "Update verified, closing before installation";
        Close();
    }

    private void OpenData(object? sender, EventArgs eventArguments)
    {
        Directory.CreateDirectory(_paths.StateRoot);
        Process.Start(new ProcessStartInfo("explorer.exe")
        {
            UseShellExecute = true,
            ArgumentList = { _paths.StateRoot },
        });
    }

    private async void BeginClosing(
        object? sender,
        FormClosingEventArgs eventArguments)
    {
        if (_allowClose)
        {
            return;
        }

        eventArguments.Cancel = true;
        if (_closing)
        {
            return;
        }

        _closing = true;
        _checkUpdates.Enabled = false;
        _status.Text = "Closing the local runtime";
        try
        {
            await RequestGracefulShutdownAsync();
            if (_backend is not null)
            {
                await _backend.StopAsync(TimeSpan.FromSeconds(30));
                await _backend.DisposeAsync();
            }
        }
        finally
        {
            _allowClose = true;
            BeginInvoke(Close);
        }
    }

    private async Task RequestGracefulShutdownAsync()
    {
        if (_webView.CoreWebView2 is null || _origin is null)
        {
            return;
        }

        const string shutdownScript =
            "(() => { const button = [...document.querySelectorAll('button')]"
            + ".find(item => item.textContent.trim() === 'Shut down');"
            + "if (!button || button.disabled) return 'unavailable';"
            + "button.click(); return 'requested'; })()";
        try
        {
            await _webView.ExecuteScriptAsync(shutdownScript);
        }
        catch (InvalidOperationException)
        {
            // The job-object fallback still guarantees process-tree cleanup.
        }
    }

    private void UpdateStartup(string message, int? percent)
    {
        if (InvokeRequired)
        {
            BeginInvoke(() => UpdateStartup(message, percent));
            return;
        }

        _startupMessage.Text = message;
        _status.Text = message;
        if (percent is null)
        {
            _startupProgress.Style = ProgressBarStyle.Marquee;
            return;
        }

        _startupProgress.Style = ProgressBarStyle.Continuous;
        _startupProgress.Value = Math.Clamp(percent.Value, 0, 100);
    }

    private static HttpClient CreateHttpClient()
    {
        HttpClientHandler handler = new()
        {
            AutomaticDecompression = DecompressionMethods.All,
            AllowAutoRedirect = true,
            MaxAutomaticRedirections = 5,
        };
        return new HttpClient(handler)
        {
            Timeout = TimeSpan.FromHours(2),
        };
    }

    private sealed class DarkColorTable : ProfessionalColorTable
    {
        public override Color ToolStripGradientBegin => Color.FromArgb(17, 27, 44);

        public override Color ToolStripGradientMiddle => Color.FromArgb(17, 27, 44);

        public override Color ToolStripGradientEnd => Color.FromArgb(17, 27, 44);

        public override Color ButtonSelectedHighlight => Color.FromArgb(32, 52, 78);

        public override Color ButtonSelectedBorder => Color.FromArgb(80, 128, 170);

        public override Color ToolStripBorder => Color.FromArgb(32, 52, 78);
    }
}
