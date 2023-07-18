using AlgorithmExtensions.Attributes;
using AlgorithmExtensions.Exceptions;
using AlgorithmExtensions.Extensions;
using Microsoft.ML;
using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using static System.Runtime.InteropServices.JavaScript.JSType;

namespace AlgorithmExtensions.Scoring
{
    public class AccuracyScoringFunction<T> : ScoringFunctionBase<T>, IScoringFunction where T : class, new()
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

            if ((predictedProperty.PropertyType != typeof(int) && predictedProperty.PropertyType != typeof(bool)) || (goldProperty.PropertyType != typeof(int) && goldProperty.PropertyType != typeof(bool)))
            {
                throw new PropertyTypeException($"Properties with attributes {typeof(GoldAttribute)} or {typeof(PredictionAttribute)} should be of type {typeof(int)} or {typeof(bool)}");
            }

            var predictedEnumerator = _mlContext.Data.CreateEnumerable<T>(predicted, true);
            var correctValues = 0f;
            var allValues = 0f;

            //var aa = predictedEnumerator.ToArray();

            foreach (var row in predictedEnumerator)
            {
                var goldValue = UnpackPropertyValue(goldProperty.GetValue(row)!);
                var predictedValue = UnpackPropertyValue(predictedProperty.GetValue(row)!);
                allValues++;
                if (goldValue == predictedValue)
                {
                    correctValues++;
                }
                else
                {
                    Debug.WriteLine("");
                }
            }

            return correctValues / allValues;
        }

        private int UnpackPropertyValue(object value)
        {
            if (value is int intVal)
            {
                return intVal;
            }
            return ((bool)value).ToInt();
        }
    }
}
