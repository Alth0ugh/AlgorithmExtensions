using AlgorithmExtensions.Attributes;
using AlgorithmExtensions.Exceptions;
using System.Reflection;

namespace AlgorithmExtensions.Scoring
{
    public class FloatScoringFunctionBase<Tout> : ScoringFunctionBase<Tout>
    {
        public void CheckIfPropertyTypeIsFloat(PropertyInfo propertyInfo)
        {
            if (propertyInfo == null)
            {
                return;
            }

            if (propertyInfo.PropertyType != typeof(float))
            {
                throw new PropertyTypeException($"Properties with attributes {typeof(GoldAttribute)} or {typeof(PredictionAttribute)} should be of type {typeof(float)}");
            }
        }
    }
}
