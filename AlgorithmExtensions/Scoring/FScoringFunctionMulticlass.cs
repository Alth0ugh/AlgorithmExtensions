﻿using Microsoft.ML;
using AlgorithmExtensions.Extensions;

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
        private int _offset;

        public FScoringFunctionMulticlass(MLContext mlContext, int classCount, bool isCountingFromOne = false, float beta = 1, FScoreType fScoreType = FScoreType.MicroAveraged)
        {
            _mlContext = mlContext;
            _beta = beta;
            _fscoreType = fScoreType;
            _classCount = classCount;
            _offset = isCountingFromOne.ToInt();
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

        private float CalculateMicroAveragedFScore(FMetricsForClass[] fMetrics)
        {
            var sum = fMetrics.Sum();

            return CalculateFScore(sum.TP, sum.FP, sum.FN);
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