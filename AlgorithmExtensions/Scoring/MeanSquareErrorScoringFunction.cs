using AlgorithmExtensions.Scoring.BaseClasses;
using Microsoft.ML;

namespace AlgorithmExtensions.Scoring
{
    /// <summary>
    /// Scoring function that calculates Mean Square Error of a regressor.
    /// </summary>
    /// <typeparam name="Tout">Object representing the structure of the output data from the model.</typeparam>
    public class MeanSquareErrorScoringFunction<Tout> : IntegerScoringFunctionBase<Tout>, IScoringFunction where Tout : class, new()
    {
        private MLContext _mlContext;
        public MeanSquareErrorScoringFunction(MLContext mlContext)
        {
            _mlContext = mlContext;
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
