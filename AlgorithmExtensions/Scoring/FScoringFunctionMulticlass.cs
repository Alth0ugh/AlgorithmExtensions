using Microsoft.ML;

namespace AlgorithmExtensions.Scoring
{
    public enum FScoreType
    {
        MacroAveraged,
        MicroAveraged
    }

    public class FScoringFunctionMulticlass<Tout> : IntegerScoringFunctionBase<Tout>, IScoringFunction where Tout : class, new()
    {
        private float _beta;
        private FScoreType _fscoreType;
        private int _classCount;
        private MLContext _mlContext;

        public FScoringFunctionMulticlass(MLContext mlContext, int classCount, float beta = 1, FScoreType fScoreType = FScoreType.MicroAveraged)
        {
            _mlContext = mlContext;
            _beta = beta;
            _fscoreType = fScoreType;
            _classCount = classCount;
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
                    fMetrics[goldValue].TP++;
                }
                else if(goldValue != predictedValue)
                {
                    fMetrics[goldValue].FN++;
                    fMetrics[predictedValue].FP++;
                }
            }

            return _fscoreType == FScoreType.MicroAveraged ? CalculateMicroAveragedFScore(fMetrics) : CalculateMacroAveragedFScore(fMetrics);
        }

        private float CalculateMicroAveragedFScore(FMetricsForClass[] fMetrics)
        {
            var tp = 0;
            var fp = 0;
            var fn = 0;

            for (int i = 0; i < fMetrics.Length; i++)
            {
                tp += fMetrics[i].TP;
                fp += fMetrics[i].FP;
                fn += fMetrics[i].FN;
            }

            return CalculateFScore(tp, fp, fn);
        }

        private float CalculateMacroAveragedFScore(FMetricsForClass[] fMetrics)
        {
            var scores = new float[fMetrics.Length];

            for (int i = 0; i < fMetrics.Length; i++)
            {
                scores[i] = CalculateFScore(fMetrics[i].TP, fMetrics[i].FP, fMetrics[i].FN);
            }

            return scores.Sum() / scores.Length;
        }

        private float CalculateFScore(int tp, int fp, int fn)
        {
            return (tp + _beta * _beta * tp) / (tp + fp + _beta * _beta * (tp + fn));
        }
    }
}
