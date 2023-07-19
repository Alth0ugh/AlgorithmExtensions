using Microsoft.ML.Trainers;

namespace AlgorithmExtensions.Hyperalgorithms
{
    public class PipelineTemplate
    {
        public List<PipelineItem> Delegates { get; } = new List<PipelineItem>();

        public void Add(Delegate creationalDelegate, string name = "")
        {
            var item = new PipelineItem(creationalDelegate, name);
            Delegates.Add(item);
        }

        public void Add(Delegate creationalDelegate, string name = "", params object[] functionParameters)
        {
            var item = new PipelineItem(creationalDelegate, name, functionParameters);
            Delegates.Add(item);
        }

        public void Add(Delegate creationalDelegate, string name = "", TrainerInputBase defaultOptions = null)
        {
            var item = new PipelineItem(creationalDelegate, name, defaultOptions: defaultOptions);
            Delegates.Add(item);
        }
    }
}
