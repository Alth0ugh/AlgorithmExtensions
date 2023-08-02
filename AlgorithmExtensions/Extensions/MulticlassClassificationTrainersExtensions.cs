using static Microsoft.ML.MulticlassClassificationCatalog;
using AlgorithmExtensions.ResNets;

namespace AlgorithmExtensions.Extensions
{
    /// <summary>
    /// Extension class for multiclass classification trainers in MLContext.
    /// </summary>
    public static class MulticlassClassificationTrainersExtensions
    {
        /// <summary>
        /// Creates untrained ResNetClassificator.
        /// </summary>
        /// <param name="trainers">Multiclass trainers catalog from MLContext.</param>
        /// <returns>Untrained ResNet.</returns>
        public static ResNetTrainer ResNetClassificator(this MulticlassClassificationTrainers trainers, Options options)
        {
            return new ResNetTrainer(options);
        }
    }
}
