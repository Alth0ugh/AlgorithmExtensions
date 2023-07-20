namespace AlgorithmExtensions.Hyperalgorithms.ParameterProviders
{
    public interface IParameterProvider
    {
        string Name { get; }
        object[] GetParameterValues();
    }
}
