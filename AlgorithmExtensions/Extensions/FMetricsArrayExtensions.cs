using AlgorithmExtensions.Scoring;

namespace AlgorithmExtensions.Extensions
{
    /// <summary>
    /// Extension for array of FMetrics.
    /// </summary>
    internal static class FMetricsArrayExtensions
    {
        /// <summary>
        /// Sums all FMetricsForClass property-wise.
        /// </summary>
        /// <param name="array">Array of FMetricsForClass.</param>
        /// <returns>FMetricsForClass representing the sum of the array.</returns>
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
