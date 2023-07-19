using AlgorithmExtensions.Scoring;

namespace AlgorithmExtensions.Extensions
{
    internal static class FMetricsArrayExtensions
    {
        internal static FMetricsForClass Sum(this FMetricsForClass[] array)
        {
            var sum = new FMetricsForClass();

            foreach ( var item in array )
            {
                sum += item;
            }

            return sum;
        }
    }
}
