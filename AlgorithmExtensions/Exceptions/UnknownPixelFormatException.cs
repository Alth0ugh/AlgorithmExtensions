namespace AlgorithmExtensions.Exceptions
{
    public class UnknownPixelFormatException : Exception
    {
        public UnknownPixelFormatException() : base() { }
        public UnknownPixelFormatException(string message) : base(message) { }
    }
}
