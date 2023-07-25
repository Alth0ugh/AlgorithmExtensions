using AlgorithmExtensions.Scoring.BaseClasses;
using Microsoft.ML;

namespace AlgorithmExtensions.Scoring
{
    /// <summary>
    /// Function that scores model's accuracy.
    /// </summary>
    /// <typeparam name="Tout">Object representing the structure of the output data from the model.</typeparam>
    public class AccuracyScoringFunction<Tout> : IntegerScoringFunctionBase<Tout>, IScoringFunction where Tout : class, new()
    {
        private MLContext _mlContext;

        public AccuracyScoringFunction(MLContext context)
        {
            _mlContext = context;
        }

        /// <inheritdoc/>
        /// <exception cref="NullReferenceException">Thrown when MLContext is null.</exception>
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
