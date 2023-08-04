namespace AlgorithmExtensions.Exceptions
{
    /// <summary>
    /// Thrown when the pixel format in a picture is unknown.
    /// </summary>
    public class UnknownPixelFormatException : Exception
    {
        public UnknownPixelFormatException() { }
        public UnknownPixelFormatException(string message) : base(message) { }
    }
}
