using AlgorithmExtensions.Attributes;
using AlgorithmExtensions.Exceptions;
using AlgorithmExtensions.Extensions;
using System.Reflection;

namespace AlgorithmExtensions.Scoring.BaseClasses
{
    /// <summary>
    /// Base class for scoring functions that predict integer numbers.
    /// </summary>
    /// <typeparam name="Tout">Object representing the structure of the output data from the model.</typeparam>
    public class IntegerScoringFunctionBase<Tout> : ScoringFunctionBase<Tout>
    {
        /// <summary>
        /// Checks if property is a number (bool or int).
        /// </summary>
        /// <param name="propertyInfo">Property to be checked.</param>
        /// <exception cref="PropertyTypeException">Thrown if the property is not a number.</exception>
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

        /// <summary>
        /// Unpacks numeric value from object.
        /// </summary>
        /// <param name="value">Value to be unpacked.</param>
        /// <returns>Unpacked numeric value.</returns>
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
