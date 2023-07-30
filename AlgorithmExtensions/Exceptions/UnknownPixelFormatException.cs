namespace AlgorithmExtensions.Exceptions
{
    public class UnknownPixelFormatException : Exception
    {
        public UnknownPixelFormatException() { }
        public UnknownPixelFormatException(string message) : base(message) { }
    }
}
