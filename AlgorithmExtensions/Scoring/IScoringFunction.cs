using Microsoft.ML;

namespace AlgorithmExtensions.Scoring
{
    /// <summary>
    /// Interface for all scoring functions.
    /// </summary>
    public interface IScoringFunction
    {
        /// <summary>
        /// Score predicted values by a model.
        /// </summary>
        /// <param name="predicted">Data containing gold data and predicted data.</param>
        /// <returns>Calculated score.</returns>
        float Score(IDataView predicted);
    }
}
