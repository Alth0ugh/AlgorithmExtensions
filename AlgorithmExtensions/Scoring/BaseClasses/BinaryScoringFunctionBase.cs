using AlgorithmExtensions.Attributes;
using AlgorithmExtensions.Exceptions;
using System.Reflection;

namespace AlgorithmExtensions.Scoring.BaseClasses
{
    /// <summary>
    /// Base class for scoring functions for binary classificators.
    /// </summary>
    /// <typeparam name="Tout">Object representing the structure of the output data from the model.</typeparam>
    public class BinaryScoringFunctionBase<Tout> : IntegerScoringFunctionBase<Tout>
    {
        /// <summary>
        /// Checks if a property is of bool type.
        /// </summary>
        /// <param name="propertyInfo">Property that is checked.</param>
        /// <exception cref="PropertyTypeException">Thrown if the property is not bool.</exception>
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
