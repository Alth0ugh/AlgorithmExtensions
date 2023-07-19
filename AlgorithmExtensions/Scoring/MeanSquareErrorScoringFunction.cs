using Microsoft.ML;

namespace AlgorithmExtensions.Scoring
{
    public class MeanSquareErrorScoringFunction<Tout> : IntegerScoringFunctionBase<Tout>, IScoringFunction where Tout : class, new()
    {
        private MLContext _mlContext;
        public MeanSquareErrorScoringFunction(MLContext mlContext)
        {
            _mlContext = mlContext;
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

            var sum = 0f;
            var count = 0;

            foreach (var row in dataEnumerator)
            {
                var goldValue = (float)goldProperty.GetValue(row);
                var predictedValue = (float)predictedProperty.GetValue(row);

                sum += (goldValue - predictedValue) * (goldValue - predictedValue);
                count++;
            }

            return sum / count;
        }
    }
}
