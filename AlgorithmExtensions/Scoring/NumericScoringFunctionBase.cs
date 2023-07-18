using AlgorithmExtensions.Attributes;
using AlgorithmExtensions.Exceptions;
using AlgorithmExtensions.Extensions;
using System.Reflection;

namespace AlgorithmExtensions.Scoring
{
    public class NumericScoringFunctionBase<Tout> : ScoringFunctionBase<Tout>
    {
        public void CheckIfPropertyTypeIsNumber(PropertyInfo propertyInfo)
        {
            if (propertyInfo == null)
            {
                return;
            }

            if (propertyInfo.PropertyType != typeof(int) && propertyInfo.PropertyType != typeof(bool))
            {
                throw new PropertyTypeException($"Properties with attributes {typeof(GoldAttribute)} or {typeof(PredictionAttribute)} should be of type {typeof(int)} or {typeof(bool)}");
            }
        }

        public int UnpackPropertyValue(object value)
        {
            if (value is int intVal)
            {
                return intVal;
            }
            return ((bool)value).ToInt();
        }
    }
}
