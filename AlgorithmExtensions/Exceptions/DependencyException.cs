namespace AlgorithmExtensions.Exceptions
{
    internal class DependencyException : Exception
    {
        public DependencyException() : base() { }
        public DependencyException(string message) : base(message) { }
        public DependencyException(string message, Exception innerException) : base(message, innerException) { }
    }
}
