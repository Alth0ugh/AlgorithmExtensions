namespace AlgorithmExtensions.Exceptions
{
    public class UnknownArchitectureException : Exception
    {
        public UnknownArchitectureException() { }
        public UnknownArchitectureException(string message) : base(message) { }
    }
}
