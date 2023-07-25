using System.CodeDom;

namespace AlgorithmExtensions.Scoring
{
    /// <summary>
    /// F-metrics used for calculation of F-score of a model.
    /// </summary>
    internal struct FMetricsForClass
    {
        /// <summary>
        /// False positive count.
        /// </summary>
        internal int FP { get; set; }
        /// <summary>
        /// True positive count.
        /// </summary>
        internal int TP { get; set; }
        /// <summary>
        /// False negative count.
        /// </summary>
        internal int FN { get; set; }

        public FMetricsForClass()
        {
            FP = 0;
            TP = 0;
            FN = 0;
        }

        public static FMetricsForClass operator +(FMetricsForClass metrics1,  FMetricsForClass metrics2)
        {
            return new FMetricsForClass()
            {
                FP = metrics1.FP + metrics2.FP,
                TP = metrics1.TP + metrics2.TP,
                FN = metrics1.FN + metrics2.FN,
            };
        }
    }
}
