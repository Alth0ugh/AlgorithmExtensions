namespace AlgorithmExtensions.Exceptions
{
    /// <summary>
    /// Thrown when parameters to be tried by GridSearch are missing.
    /// </summary>
    public class ParametersMissingException : Exception
    {
        public ParametersMissingException() { }
        public ParametersMissingException(string message) : base(message) { }
    }
}
