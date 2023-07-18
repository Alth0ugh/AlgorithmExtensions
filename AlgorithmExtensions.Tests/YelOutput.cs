using AlgorithmExtensions.Attributes;
using Microsoft.ML.Data;

namespace AlgorithmExtensions.Tests
{
    internal class YelOutput : YelpInput
    {
        [Prediction]
        [ColumnName("PredictedLabel")]
        public bool Prediction { get; set; }

        public float Score { get; set; }
    }
}