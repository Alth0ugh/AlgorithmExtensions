using AlgorithmExtensions.Scoring.BaseClasses;
using Microsoft.ML;

namespace AlgorithmExtensions.Scoring
{
    /// <summary>
    /// Function that calculates model's F-score for binary classification.
    /// </summary>
    /// <typeparam name="Tout">Object representing the structure of the output data from the model.</typeparam>
    public class FScoringFunctionBinary<Tout> : BinaryScoringFunctionBase<Tout>, IScoringFunction where Tout : class, new()
    {
        private MLContext _mlContext;
        private float _beta;
        public FScoringFunctionBinary(MLContext mlContext, float beta = 1)
        {
            _mlContext = mlContext;
            _beta = beta;
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
            var fp = 0;
            var fn = 0;
            var tp = 0;

            foreach (var row in dataEnumerator)
            {
                var goldValue = (bool)goldProperty.GetValue(row)!;
                var predictedValue = (bool)predictedProperty.GetValue(row)!;

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
