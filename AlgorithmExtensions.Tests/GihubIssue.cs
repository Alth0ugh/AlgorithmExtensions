using Microsoft.ML.Data;

namespace AlgorithmExtensions.Tests
{
    public class GihubIssue
    {
        [LoadColumn(0)]
        public string? ID { get; set; }
        [LoadColumn(1)]
        public string? Area { get; set; }
        [LoadColumn(2)]
        public string? Title { get; set; }
        [LoadColumn(3)]
        public string? Description { get; set; }
    }
}
