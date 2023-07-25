using AlgorithmExtensions.Attributes;
using AlgorithmExtensions.Exceptions;
using System.Reflection;

namespace AlgorithmExtensions.Scoring.BaseClasses
{
    /// <summary>
    /// Base class for scoring functions for regressors that predict float values.
    /// </summary>
    /// <typeparam name="Tout">Object representing the structure of the output data from the model.</typeparam>
    public class FloatScoringFunctionBase<Tout> : ScoringFunctionBase<Tout>
    {
        /// <summary>
        /// Checks if the property is float.
        /// </summary>
        /// <param name="propertyInfo">Property that is checked.</param>
        /// <exception cref="PropertyTypeException">Thrown if the property is not float.</exception>
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
