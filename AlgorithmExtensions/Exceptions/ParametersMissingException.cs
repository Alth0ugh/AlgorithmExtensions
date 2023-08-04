namespace AlgorithmExtensions.Exceptions
{
    public class ParametersMissingException : Exception
    {
        public ParametersMissingException() { }
        public ParametersMissingException(string message) : base(message) { }
    }
}
