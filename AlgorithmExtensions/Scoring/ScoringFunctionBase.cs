using AlgorithmExtensions.Attributes;
using AlgorithmExtensions.Exceptions;
using System.Reflection;

namespace AlgorithmExtensions.Scoring
{
    public class ScoringFunctionBase<T>
    {
        public PropertyInfo GetGoldProperty()
        {
            var goldProperties = from prop in typeof(T).GetProperties()
                                 where prop.IsDefined(typeof(GoldAttribute), true)
                                 select prop;

            var goldPropertiesCount = goldProperties.Count();

            if (goldPropertiesCount != 1)
            {
                throw new MissingAttributeException(goldPropertiesCount == 0 ? $"Type {typeof(T)} is missing attribute {typeof(GoldAttribute)}" :
                    $"Type {typeof(T)} has multiple {typeof(GoldAttribute)} attributes");
            }
            return goldProperties.First();
        }

        public PropertyInfo GetPredictionProperty()
        {
            var predictedProperties = from prop in typeof(T).GetProperties()
                                      where prop.IsDefined(typeof(PredictionAttribute), true)
                                      select prop;

            var predictedPropertiesCount = predictedProperties.Count();

            if (predictedPropertiesCount != 1)
            {
                throw new MissingAttributeException(predictedPropertiesCount == 0 ? $"Type {typeof(T)} is missing attribute {typeof(PredictionAttribute)}" :
                    $"Type {typeof(T)} has multiple {typeof(PredictionAttribute)} attributes");
            }

            return predictedProperties.First();
        }
    }
}
