using Microsoft.ML;

namespace AlgorithmExtensions.Scoring
{
    public class AccuracyScoringFunction<Tout> : IntegerScoringFunctionBase<Tout>, IScoringFunction where Tout : class, new()
    {
        private MLContext _mlContext;
        public AccuracyScoringFunction(MLContext context)
        {
            _mlContext = context;
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

            var predictedEnumerator = _mlContext.Data.CreateEnumerable<Tout>(predicted, true);
            var correctValues = 0f;
            var allValues = 0f;

            foreach (var row in predictedEnumerator)
            {
                var goldValue = UnpackPropertyValue(goldProperty.GetValue(row)!);
                var predictedValue = UnpackPropertyValue(predictedProperty.GetValue(row)!);
                allValues++;
                if (goldValue == predictedValue)
                {
                    correctValues++;
                }
            }

            return correctValues / allValues;
        }
    }
}
