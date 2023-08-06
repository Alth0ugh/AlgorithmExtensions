using AlgorithmExtensions.Attributes;

namespace AlgorithmExtensions.Tests
{
    public class TaxiPrediction : TaxiTrip
    {
        [Prediction]
        public float Score { get; set; }
    }
}
