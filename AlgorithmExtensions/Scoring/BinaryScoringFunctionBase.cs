using AlgorithmExtensions.Attributes;
using AlgorithmExtensions.Exceptions;
using System.Reflection;

namespace AlgorithmExtensions.Scoring
{
    public class BinaryScoringFunctionBase<Tout> : IntegerScoringFunctionBase<Tout>
    {
        public void CheckIfPropertyTypeIsBool(PropertyInfo propertyInfo)
        {
            if (propertyInfo == null)
            {
                return;
            }

            if (propertyInfo.PropertyType != typeof(bool))
            {
                throw new PropertyTypeException($"Properties with attributes {typeof(GoldAttribute)} or {typeof(PredictionAttribute)} should be of type {typeof(bool)}");
            }
        }
    }
}
