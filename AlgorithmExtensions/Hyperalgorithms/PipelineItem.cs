using Microsoft.ML;
using Microsoft.ML.Trainers;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using static Microsoft.ML.TrainCatalogBase;

namespace AlgorithmExtensions.Hyperalgorithms
{
    public class PipelineItem
    {
        public Delegate CreationalDelegate { get; set; }
        public string Name { get; set; }
        public object[] DefaultParameters { get; set; }
        public TrainerInputBase DefaultOptions { get; set; }
        public PipelineItem()
        {
            
        }

        public PipelineItem(Delegate creationalDelegate, string name = "", object[] defaultParameters = null, TrainerInputBase defaultOptions = null)
        {
            CreationalDelegate = creationalDelegate;
            Name = name;
            DefaultParameters = defaultParameters;
            DefaultOptions = defaultOptions;
        }
    }
}
