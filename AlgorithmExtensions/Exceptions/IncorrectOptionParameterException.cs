namespace AlgorithmExtensions.Exceptions
{
    public class IncorrectOptionParameterException : Exception
    {
        public IncorrectOptionParameterException() : base() { }
        public IncorrectOptionParameterException(string message) : base(message) { }
    }
}
