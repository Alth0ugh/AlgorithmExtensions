namespace AlgorithmExtensions.Hyperalgorithms.ParameterProviders
{
    /// <summary>
    /// Provides constant values of parameters to GridSearchCV.
    /// </summary>
    public class ConstantParameterProvider : IParameterProvider
    {
        /// <inheritdoc/>
        public string Name { get; set; }
        /// <summary>
        /// Parameter values.
        /// </summary>
        private object[] _parameterValues { get; set; }

        /// <summary>
        /// Creates new instance of the provider.
        /// </summary>
        /// <param name="parameterName">Name of the parameter that is being provided with the value.</param>
        /// <param name="parameterValues">Values to be provided to the parameter.</param>
        public ConstantParameterProvider(string parameterName, params object[] parameterValues)
        {
            Name = parameterName;
            _parameterValues = parameterValues;
        }

        /// <inheritdoc/>
        public object[] GetParameterValues()
        {
            return _parameterValues;
        }
    }
}
