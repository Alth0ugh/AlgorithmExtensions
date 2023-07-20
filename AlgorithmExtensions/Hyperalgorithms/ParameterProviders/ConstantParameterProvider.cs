namespace AlgorithmExtensions.Hyperalgorithms.ParameterProviders
{
    public class ConstantParameterProvider : IParameterProvider
    {
        public string Name { get; set; }
        public object[] ParameterValues { get; set; }

        public ConstantParameterProvider(string parameterName, params object[] parameterValues)
        {
            Name = parameterName;
            ParameterValues = parameterValues;
        }

        public object[] GetParameterValues()
        {
            return ParameterValues;
        }
    }
}
