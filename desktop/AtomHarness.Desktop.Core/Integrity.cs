using System.Security.Cryptography;
using System.Text.RegularExpressions;

namespace LucernaLabs.AtomHarness.Desktop;

public static partial class Integrity
{
    [GeneratedRegex("^[0-9a-f]{64}$", RegexOptions.CultureInvariant)]
    private static partial Regex Sha256Pattern();

    [GeneratedRegex(
        "^(con|prn|aux|nul|com[1-9]|lpt[1-9])$",
        RegexOptions.CultureInvariant | RegexOptions.IgnoreCase)]
    private static partial Regex WindowsDeviceNamePattern();

    public static bool IsSha256(string? value)
    {
        return value is not null && Sha256Pattern().IsMatch(value);
    }

    public static async Task<string> Sha256FileAsync(
        string path,
        CancellationToken cancellationToken = default)
    {
        await using FileStream stream = new(
            path,
            FileMode.Open,
            FileAccess.Read,
            FileShare.Read,
            1024 * 1024,
            FileOptions.Asynchronous | FileOptions.SequentialScan);
        byte[] digest = await SHA256.HashDataAsync(stream, cancellationToken);
        return Convert.ToHexStringLower(digest);
    }

    public static bool IsWithin(string candidate, string root)
    {
        string fullCandidate = Path.GetFullPath(candidate);
        string fullRoot = Path.GetFullPath(root)
            .TrimEnd(Path.DirectorySeparatorChar, Path.AltDirectorySeparatorChar)
            + Path.DirectorySeparatorChar;
        return fullCandidate.StartsWith(
            fullRoot,
            StringComparison.OrdinalIgnoreCase);
    }

    public static bool IsSafeRelativePath(string? value)
    {
        if (string.IsNullOrWhiteSpace(value)
            || Path.IsPathRooted(value))
        {
            return false;
        }

        string normalized = value.Replace('\\', '/');
        string[] parts = normalized.Split('/');
        foreach (string part in parts)
        {
            if (part is "" or "." or ".."
                || part.EndsWith(' ')
                || part.EndsWith('.')
                || part.IndexOfAny(Path.GetInvalidFileNameChars()) >= 0)
            {
                return false;
            }

            string deviceCandidate = part.Split('.')[0].TrimEnd(' ', '.');
            if (WindowsDeviceNamePattern().IsMatch(deviceCandidate))
            {
                return false;
            }
        }

        return true;
    }
}
