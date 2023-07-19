using AlgorithmExtensions.Attributes;
using AlgorithmExtensions.Exceptions;
using AlgorithmExtensions.Extensions;
using System.Reflection;

namespace AlgorithmExtensions.Scoring
{
    public class IntegerScoringFunctionBase<Tout> : ScoringFunctionBase<Tout>
    {
        public void CheckIfPropertyTypeIsNumber(PropertyInfo propertyInfo)
        {
            if (propertyInfo == null)
            {
                return;
            }

            if (propertyInfo.PropertyType != typeof(int) && propertyInfo.PropertyType != typeof(bool) && propertyInfo.PropertyType != typeof(uint))
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
            else if (value is uint uintVal)
            {
                return (int)uintVal;
            }
            return ((bool)value).ToInt();
        }
    }
}
