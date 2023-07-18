using Microsoft.ML;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace AlgorithmExtensions.Scoring
{
    public interface IScoringFunction
    {
        float Score(IDataView predicted);
    }
}
