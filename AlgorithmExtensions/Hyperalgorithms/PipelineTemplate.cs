using Microsoft.ML.Trainers;

namespace AlgorithmExtensions.Hyperalgorithms
{
    /// <summary>
    /// Represents a machine learning pipeline.
    /// </summary>
    public class PipelineTemplate
    {
        /// <summary>
        /// List of items in the pipeline
        /// </summary>
        public List<PipelineItem> Items { get; } = new List<PipelineItem>();

        /// <summary>
        /// Adds a creational delegate to the pipeline.
        /// </summary>
        /// <param name="creationalDelegate">Creational delegate to be added.</param>
        /// <param name="name">Name to be associated with result of calling of the creational delegate.</param>
        public void Add(Delegate creationalDelegate, string name)
        {
            var item = new PipelineItem(creationalDelegate, name);
            Items.Add(item);
        }

        /// <summary>
        /// Adds a creational delegate to the pipeline.
        /// </summary>
        /// <param name="creationalDelegate">Creational delegate to be added.</param>
        /// <param name="name">Name to be associated with result of calling of the creational delegate.</param>
        /// <param name="functionParameters">Parameters to be passed to the creational delegate.</param>
        public void Add(Delegate creationalDelegate, string name, params object[] functionParameters)
        {
            var item = new PipelineItem(creationalDelegate, name, functionParameters);
            Items.Add(item);
        }

        /// <summary>
        /// Adds a creational delegate to the pipeline.
        /// </summary>
        /// <param name="creationalDelegate">Creational delegate to be added.</param>
        /// <param name="name">Name to be associated with result of calling of the creational delegate.</param>
        /// <param name="functionParameters">Parameters to be passed to the creational delegate.</param>
        public void Add(Delegate creationalDelegate, string name, object defaultOptions)
        {
            var item = new PipelineItem(creationalDelegate, name, defaultOptions: defaultOptions);
            Items.Add(item);
        }
    }
}
