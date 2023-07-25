using AlgorithmExtensions.Attributes;
using AlgorithmExtensions.Exceptions;
using System.Reflection;

namespace AlgorithmExtensions.Scoring.BaseClasses
{
    /// <summary>
    /// Base class for scoring functions.
    /// </summary>
    /// <typeparam name="Tout">Object representing the structure of the output data from the model.</typeparam>
    public class ScoringFunctionBase<Tout>
    {
        /// <summary>
        /// Retrieves property containing GoldAttribute from output data object.
        /// </summary>
        /// <returns>Property info for the retrieved property.</returns>
        /// <exception cref="MissingAttributeException">Thrown if no property decorated with GoldAttributes is found.</exception>
        public PropertyInfo GetGoldProperty()
        {
            var goldProperties = from prop in typeof(Tout).GetProperties()
                                 where prop.IsDefined(typeof(GoldAttribute), true)
                                 select prop;

            var goldPropertiesCount = goldProperties.Count();

            if (goldPropertiesCount != 1)
            {
                throw new MissingAttributeException(goldPropertiesCount == 0 ? $"Type {typeof(Tout)} is missing attribute {typeof(GoldAttribute)}" :
                    $"Type {typeof(Tout)} has multiple {typeof(GoldAttribute)} attributes");
            }
            return goldProperties.First();
        }

        /// <summary>
        /// Retrieves property containing PredictionAttribute from output data object.
        /// </summary>
        /// <returns>Property info for the retrieved property.</returns>
        /// <exception cref="MissingAttributeException">Thrown if no property decorated with GoldAttributes is found.</exception>
        public PropertyInfo GetPredictionProperty()
        {
            var predictedProperties = from prop in typeof(Tout).GetProperties()
                                      where prop.IsDefined(typeof(PredictionAttribute), true)
                                      select prop;

            var predictedPropertiesCount = predictedProperties.Count();

            if (predictedPropertiesCount != 1)
            {
                throw new MissingAttributeException(predictedPropertiesCount == 0 ? $"Type {typeof(Tout)} is missing attribute {typeof(PredictionAttribute)}" :
                    $"Type {typeof(Tout)} has multiple {typeof(PredictionAttribute)} attributes");
            }

            return predictedProperties.First();
        }
    }
}
