using Microsoft.ML;
using AlgorithmExtensions.Extensions;
using AlgorithmExtensions.Scoring.BaseClasses;

namespace AlgorithmExtensions.Scoring
{
    /// <summary>
    /// Type of F-score to calculate in multiclass classification.
    /// </summary>
    public enum FScoreType
    {
        MacroAveraged,
        MicroAveraged
    }

    /// <summary>
    /// Function that calculates model's F-score for multiclass classification.
    /// </summary>
    /// <typeparam name="Tout">Object representing the structure of the output data from the model.</typeparam>
    public class FScoringFunctionMulticlass<Tout> : IntegerScoringFunctionBase<Tout>, IScoringFunction where Tout : class, new()
    {
        private float _beta;
        private FScoreType _fscoreType;
        private int _classCount;
        private MLContext _mlContext;
        private int _offset;

        /// <summary>
        /// Creates new instance of F-score calculator.
        /// </summary>
        /// <param name="mlContext">Machine learning context.</param>
        /// <param name="classCount">Number of classes present in the data.</param>
        /// <param name="isCountingFromOne">True if class numbers start from 1, otherwise false and classes start from 0.</param>
        /// <param name="beta">Beta parameter for F-score.</param>
        /// <param name="fScoreType">Type of multiclass F-score calculation.</param>
        public FScoringFunctionMulticlass(MLContext mlContext, int classCount, bool isCountingFromOne = false, float beta = 1, FScoreType fScoreType = FScoreType.MicroAveraged)
        {
            _mlContext = mlContext;
            _beta = beta;
            _fscoreType = fScoreType;
            _classCount = classCount;
            _offset = isCountingFromOne.ToInt();
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
            var fMetrics = new FMetricsForClass[_classCount];

            for (int i = 0; i < fMetrics.Length; i++)
            {
                fMetrics[i] = new FMetricsForClass();
            }

            foreach (var row in dataEnumerator)
            {
                var goldValue = UnpackPropertyValue(goldProperty.GetValue(row));
                var predictedValue = UnpackPropertyValue(predictedProperty.GetValue(row));

                if (goldValue == predictedValue)
                {
                    fMetrics[goldValue - _offset].TP++;
                }
                else if(goldValue != predictedValue)
                {
                    fMetrics[goldValue - _offset].FN++;
                    fMetrics[predictedValue - _offset].FP++;//Throw exception when there are more classes than supplied in the parameter
                }
            }

            return _fscoreType == FScoreType.MicroAveraged ? CalculateMicroAveragedFScore(fMetrics) : CalculateMacroAveragedFScore(fMetrics);
        }

        /// <summary>
        /// Calculates micro-averaged F-score.
        /// </summary>
        /// <param name="fMetrics">Array of measured F-metrics.</param>
        /// <returns>Resulting F-score.</returns>
        private float CalculateMicroAveragedFScore(FMetricsForClass[] fMetrics)
        {
            var sum = fMetrics.Sum();

            return CalculateFScore(sum.TP, sum.FP, sum.FN);
        }

        /// <summary>
        /// Calculates macro-averaged F-score.
        /// </summary>
        /// <param name="fMetrics">Array of measured F-metrics.</param>
        /// <returns>Resulting F-score.</returns>
        private float CalculateMacroAveragedFScore(FMetricsForClass[] fMetrics)
        {
            var scores = new float[fMetrics.Length];

            for (int i = 0; i < fMetrics.Length; i++)
            {
                scores[i] = CalculateFScore(fMetrics[i].TP, fMetrics[i].FP, fMetrics[i].FN);
            }

            return scores.Sum() / scores.Length;
        }

        /// <summary>
        /// Calculates F-score.
        /// </summary>
        /// <param name="tp">True positive count.</param>
        /// <param name="fp">False positive count.</param>
        /// <param name="fn">False negative count.</param>
        /// <returns>Resultinig F-score.</returns>
        private float CalculateFScore(int tp, int fp, int fn)
        {
            return (tp + _beta * _beta * tp) / (tp + fp + _beta * _beta * (tp + fn));
        }
    }
}
