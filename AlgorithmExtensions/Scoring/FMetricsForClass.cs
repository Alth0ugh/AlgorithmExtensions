using System.CodeDom;

namespace AlgorithmExtensions.Scoring
{
    internal struct FMetricsForClass
    {
        internal int FP { get; set; }
        internal int TP { get; set; }
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
