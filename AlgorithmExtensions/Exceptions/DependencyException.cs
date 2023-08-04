namespace AlgorithmExtensions.Exceptions
{
    /// <summary>
    /// Thrown when dependend library for operation raises an exception.
    /// </summary>
    public class DependencyException : Exception
    {
        public DependencyException() : base() { }
        public DependencyException(string message) : base(message) { }
        public DependencyException(string message, Exception innerException) : base(message, innerException) { }
    }
}
