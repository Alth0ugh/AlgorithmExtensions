namespace AlgorithmExtensions.Exceptions
{
    /// <summary>
    /// Thrown when the ResNet architecture is unknown.
    /// </summary>
    public class UnknownArchitectureException : Exception
    {
        public UnknownArchitectureException() { }
        public UnknownArchitectureException(string message) : base(message) { }
    }
}
