using LucernaLabs.AtomHarness.Desktop;

namespace LucernaLabs.AtomHarness.Desktop.Tests;

[TestClass]
public sealed class CertifiedModelContractTests
{
    [TestMethod]
    public void CertifiedModelRequiresExactSizeHashAndHttps()
    {
        string root = TestFiles.CreateRoot();
        try
        {
            string path = Path.Combine(root, "atom-language-model.json");
            File.WriteAllText(
                path,
                """
                {
                  "schema": 1,
                  "runtime": "atom-language-model-contract-v1",
                  "role": "language-only-membrane",
                  "artifact": {
                    "filename": "qwen3-4b-instruct-2507-q8_0.gguf",
                    "bytes": 4280403520,
                    "sha256": "ae916ede1c010a26955ee8ae2e908bf8815a3f135ec860439ab924701c69d5f1",
                    "download_url": "https://models.example/qwen.gguf",
                    "default_relative_path": "../models/qwen.gguf"
                  }
                }
                """);
            CertifiedModelContract contract = CertifiedModelContract.Load(path);
            Assert.AreEqual(4_280_403_520, contract.Artifact.Bytes);
            Assert.AreEqual(
                "ae916ede1c010a26955ee8ae2e908bf8815a3f135ec860439ab924701c69d5f1",
                contract.Artifact.Sha256);
        }
        finally
        {
            Directory.Delete(root, true);
        }
    }
}
