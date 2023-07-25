namespace AlgorithmExtensions.Extensions
{
    /// <summary>
    /// Extension class for bool type.
    /// </summary>
    internal static class BoolExtensions
    {
        /// <summary>
        /// Converts bool value into integer value.
        /// </summary>
        /// <param name="value">Bool value to be converted.</param>
        /// <returns>Converted bool value into integer.</returns>
        internal static int ToInt(this bool value)
        {
            if (value)
                return 1;
            return 0;
        }
    }
}
