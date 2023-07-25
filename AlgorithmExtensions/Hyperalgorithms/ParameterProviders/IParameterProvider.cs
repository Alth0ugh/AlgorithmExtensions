namespace AlgorithmExtensions.Hyperalgorithms.ParameterProviders
{
    /// <summary>
    /// Represents provider of values for a given parameter.
    /// </summary>
    public interface IParameterProvider
    {
        /// <summary>
        /// Name of the parameter that the provider provides values to.
        /// </summary>
        string Name { get; }
        /// <summary>
        /// Generates all of the values for the parameter.
        /// </summary>
        /// <returns>Array of the values for the parameter.</returns>
        object[] GetParameterValues();
    }
}
