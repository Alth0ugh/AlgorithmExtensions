using Microsoft.ML;

namespace AlgorithmExtensions.Scoring
{
    public interface IScoringFunction
    {
        float Score(IDataView predicted);
    }
}
