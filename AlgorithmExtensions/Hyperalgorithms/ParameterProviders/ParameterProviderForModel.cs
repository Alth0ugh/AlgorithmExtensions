namespace AlgorithmExtensions.Hyperalgorithms.ParameterProviders
{
    /// <summary>
    /// Provides all parameter values for a model.
    /// </summary>
    public class ParameterProviderForModel : Dictionary<string, IParameterProvider[]>
    {
        /// <summary>
        /// Adds parameter providers for a model.
        /// </summary>
        /// <param name="name">Model name.</param>
        /// <param name="parameterProviders">Parameter providers for the model.</param>
        public new void Add(string name, params IParameterProvider[] parameterProviders)
        {
            base.Add(name, parameterProviders);
        }
    }
}
