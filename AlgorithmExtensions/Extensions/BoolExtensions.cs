namespace AlgorithmExtensions.Extensions
{
    internal static class BoolExtensions
    {
        internal static int ToInt(this bool value)
        {
            if (value)
                return 1;
            return 0;
        }
    }
}
