namespace AlgorithmExtensions.Hyperalgorithms.ParameterProviders
{
    public class ParameterProviderForModel : Dictionary<string, IParameterProvider[]>
    {
        public string Name { get; set; }
        public List<IParameterProvider> ModelParameters { get; set; }

        public void Add(string name, params IParameterProvider[] parameterProviders)
        {
            base.Add(name, parameterProviders);
        }
    }
}
