using Microsoft.ML.Trainers;

namespace AlgorithmExtensions.Hyperalgorithms
{
    /// <summary>
    /// Represents an item in mechine learning pipeline.
    /// </summary>
    public class PipelineItem
    {
        /// <summary>
        /// Delegate to be used to create an estimator or a transformer.
        /// </summary>
        public Delegate CreationalDelegate { get; set; }
        /// <summary>
        /// Name to be associated with the estimator or transformer.
        /// </summary>
        public string Name { get; set; }
        /// <summary>
        /// Default parameters to be used in the creational delegate for transformer.
        /// </summary>
        public object[]? DefaultParameters { get; set; }
        /// <summary>
        /// Default options for estimator.
        /// </summary>
        public object? DefaultOptions { get; set; }

        /// <summary>
        /// Creates new instance of PipelineItem.
        /// </summary>
        /// <param name="creationalDelegate">Creational delegate for generating the item.</param>
        /// <param name="name">Name assiciated with the item.</param>
        /// <param name="defaultParameters">Default parameters if the item is transformer.</param>
        /// <param name="defaultOptions">Default options if the item is estimator.</param>
        public PipelineItem(Delegate creationalDelegate, string name = "", object[]? defaultParameters = null, object? defaultOptions = null)
        {
            CreationalDelegate = creationalDelegate;
            Name = name;
            DefaultParameters = defaultParameters;
            DefaultOptions = defaultOptions;
        }
    }
}
