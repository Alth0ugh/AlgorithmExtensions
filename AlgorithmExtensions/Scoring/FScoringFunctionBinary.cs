using Microsoft.ML;

namespace AlgorithmExtensions.Scoring
{
    public class FScoringFunctionBinary<Tout> : BinaryScoringFunctionBase<Tout>, IScoringFunction where Tout : class, new()
    {
        private MLContext _mlContext;
        private float _beta;
        public FScoringFunctionBinary(MLContext mlContext, float beta = 1)
        {
            _mlContext = mlContext;
            _beta = beta;
        }

        public float Score(IDataView predicted)
        {
            if (_mlContext == null)
            {
                throw new NullReferenceException($"No instance of {typeof(MLContext)} was passed");
            }

            var predictedProperty = GetPredictionProperty();
            var goldProperty = GetGoldProperty();

            CheckIfPropertyTypeIsNumber(predictedProperty);
            CheckIfPropertyTypeIsNumber(goldProperty);

            var dataEnumerator = _mlContext.Data.CreateEnumerable<Tout>(predicted, true);
            var fp = 0;
            var fn = 0;
            var tp = 0;

            foreach (var row in dataEnumerator)
            {
                var goldValue = (bool)goldProperty.GetValue(row);
                var predictedValue = (bool)predictedProperty.GetValue(row);

                if (predictedValue && goldValue)
                {
                    tp++;
                }
                else if (predictedValue && !goldValue)
                {
                    fp++;
                }
                else if (!predictedValue && goldValue)
                {
                    fn++;
                }
            }

            return (tp + _beta * _beta * tp) / (tp + fp + _beta * _beta * (tp + fn));
        }
    }
}
