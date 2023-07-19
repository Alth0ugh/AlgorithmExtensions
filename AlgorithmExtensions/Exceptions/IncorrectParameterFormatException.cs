namespace AlgorithmExtensions.Exceptions
{
    public class IncorrectParameterFormatException : Exception
    {
        public IncorrectParameterFormatException() : base() { }
        public IncorrectParameterFormatException(string message) : base(message) { }
    }
}
