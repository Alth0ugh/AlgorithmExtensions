using Microsoft.ML;
using Microsoft.ML.Trainers;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Linq.Expressions;
using System.Text;
using System.Threading.Tasks;
using static Microsoft.ML.TrainCatalogBase;

namespace AlgorithmExtensions.Hyperalgorithms
{
    public class PipelineTemplate
    {
        public List<PipelineItem> Delegates { get; } = new List<PipelineItem>();

        public void Add(Delegate creationalDelegate, CatalogInstantiatorBase catalog, string name = "")
        {
            var item = new PipelineItem(creationalDelegate, catalog, name);
            Delegates.Add(item);
        }

        public void Add(Delegate creationalDelegate, TransformsCatalog catalog, string name = "")
        {
            var item = new PipelineItem(creationalDelegate, catalog, name);
            Delegates.Add(item);
        }
    }
}
