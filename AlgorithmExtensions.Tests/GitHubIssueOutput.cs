using AlgorithmExtensions.Attributes;

namespace AlgorithmExtensions.Tests
{
    public class GitHubIssueOutput : GihubIssue
    {
        [Gold]
        public uint Label { get; set; }
        [Prediction]
        public uint PredictedLabel { get; set; }
    }
}
