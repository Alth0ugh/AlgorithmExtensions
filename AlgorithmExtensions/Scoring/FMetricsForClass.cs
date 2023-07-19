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
    }
}
