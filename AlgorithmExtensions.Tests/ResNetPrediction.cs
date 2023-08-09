using AlgorithmExtensions.Attributes;

namespace AlgorithmExtensions.Tests
{
    public class ResNetPrediction
    {
        [Gold]
        public uint Prediction { get; set; }
        [Prediction]
        public uint Gold { get; set; }
    }
}
