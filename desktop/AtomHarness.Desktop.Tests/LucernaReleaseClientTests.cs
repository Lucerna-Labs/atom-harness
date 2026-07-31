using System.Net;
using System.Security.Cryptography;
using System.Text;
using LucernaLabs.AtomHarness.Desktop;

namespace LucernaLabs.AtomHarness.Desktop.Tests;

[TestClass]
public sealed class LucernaReleaseClientTests
{
    [TestMethod]
    public async Task NewerFeedDownloadsOnlyVerifiedArtifact()
    {
        string root = TestFiles.CreateRoot();
        try
        {
            byte[] artifact = Encoding.UTF8.GetBytes("verified-update");
            string digest = Convert.ToHexStringLower(SHA256.HashData(artifact));
            string feed = $$"""
                {
                  "schema": 1,
                  "app_id": "com.lucernalabs.atom-harness",
                  "platform": "windows-x64",
                  "version": "5.1.0",
                  "release_notes": "A verified test update.",
                  "artifact": {
                    "url": "https://updates.example/atom-harness.zip",
                    "bytes": {{artifact.Length}},
                    "sha256": "{{digest}}"
                  }
                }
                """;
            using HttpClient httpClient = new(
                new StaticHandler(feed, artifact));
            UpdatePolicyManifest manifest = TestFiles.LoadUpdateContract(
                root,
                "5.0.0");
            LucernaReleaseClient client = new(
                httpClient,
                manifest,
                Path.Combine(root, "staging"));
            UpdateOffer? offer = await client.CheckForUpdateAsync();
            Assert.IsNotNull(offer);
            string path = await client.DownloadAndVerifyAsync(offer);
            Assert.AreEqual(digest, await Integrity.Sha256FileAsync(path));
        }
        finally
        {
            Directory.Delete(root, true);
        }
    }

    [TestMethod]
    public async Task TamperedArtifactIsDeletedAndRejected()
    {
        string root = TestFiles.CreateRoot();
        try
        {
            byte[] artifact = Encoding.UTF8.GetBytes("tampered-update");
            string feed = $$"""
                {
                  "schema": 1,
                  "app_id": "com.lucernalabs.atom-harness",
                  "platform": "windows-x64",
                  "version": "5.1.0",
                  "release_notes": "A tampered test update.",
                  "artifact": {
                    "url": "https://updates.example/atom-harness.zip",
                    "bytes": {{artifact.Length}},
                    "sha256": "{{new string('0', 64)}}"
                  }
                }
                """;
            using HttpClient httpClient = new(
                new StaticHandler(feed, artifact));
            UpdatePolicyManifest manifest = TestFiles.LoadUpdateContract(
                root,
                "5.0.0");
            string staging = Path.Combine(root, "staging");
            LucernaReleaseClient client = new(httpClient, manifest, staging);
            UpdateOffer? offer = await client.CheckForUpdateAsync();
            Assert.IsNotNull(offer);
            await Assert.ThrowsExactlyAsync<CryptographicException>(
                () => client.DownloadAndVerifyAsync(offer));
            Assert.AreEqual(0, Directory.GetFiles(staging).Length);
        }
        finally
        {
            Directory.Delete(root, true);
        }
    }

    private sealed class StaticHandler(
        string feed,
        byte[] artifact) : HttpMessageHandler
    {
        protected override Task<HttpResponseMessage> SendAsync(
            HttpRequestMessage request,
            CancellationToken cancellationToken)
        {
            cancellationToken.ThrowIfCancellationRequested();
            HttpContent content = request.RequestUri?.AbsolutePath.EndsWith(
                ".json",
                StringComparison.Ordinal) == true
                ? new StringContent(feed, Encoding.UTF8, "application/json")
                : new ByteArrayContent(artifact);
            content.Headers.ContentLength = content is ByteArrayContent
                ? artifact.Length
                : content.Headers.ContentLength;
            return Task.FromResult(new HttpResponseMessage(HttpStatusCode.OK)
            {
                RequestMessage = request,
                Content = content,
            });
        }
    }
}
