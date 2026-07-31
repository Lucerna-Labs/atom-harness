using LucernaLabs.AtomHarness.Desktop;

namespace LucernaLabs.AtomHarness.Desktop.Tests;

[TestClass]
public sealed class UpdateContractTests
{
    [TestMethod]
    public void ValidContractRequiresExplicitVerifiedUpdates()
    {
        string root = TestFiles.CreateRoot();
        try
        {
            string path = Path.Combine(root, "lucerna-update.json");
            File.WriteAllText(path, TestFiles.UpdateContractJson("5.0.0"));
            UpdatePolicyManifest manifest = UpdatePolicyManifest.Load(path);
            Assert.AreEqual("lucerna-release-client-v1", manifest.Runtime);
            Assert.IsTrue(manifest.Policy.ExplicitUserConsentRequired);
            Assert.IsTrue(manifest.Policy.ArtifactSha256Required);
            Assert.IsFalse(manifest.Policy.AutomaticInstall);
        }
        finally
        {
            Directory.Delete(root, true);
        }
    }

    [TestMethod]
    public void ContractRejectsInsecureFeed()
    {
        string root = TestFiles.CreateRoot();
        try
        {
            string path = Path.Combine(root, "lucerna-update.json");
            File.WriteAllText(
                path,
                TestFiles.UpdateContractJson("5.0.0")
                    .Replace("https://updates.example", "http://updates.example"));
            Assert.ThrowsExactly<InvalidDataException>(
                () => UpdatePolicyManifest.Load(path));
        }
        finally
        {
            Directory.Delete(root, true);
        }
    }

    [TestMethod]
    public void StableVersionOrdersReleases()
    {
        StableVersion current = StableVersion.Parse("5.0.0");
        StableVersion next = StableVersion.Parse("5.1.0");
        Assert.IsTrue(next.CompareTo(current) > 0);
        Assert.ThrowsExactly<InvalidDataException>(
            () => StableVersion.Parse("5.1"));
    }
}
