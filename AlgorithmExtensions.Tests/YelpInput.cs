using AlgorithmExtensions.Attributes;
using Microsoft.ML.Data;

namespace AlgorithmExtensions.Tests
{
    public class YelpInput
    {
        [LoadColumn(0)]
        public string? Text { get; set; }
        [LoadColumn(1)]
        [Gold]
        public bool Label { get; set; }
    }
}