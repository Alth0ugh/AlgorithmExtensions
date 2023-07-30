namespace AlgorithmExtensions.Exceptions
{
    /// <summary>
    /// Thrown when two variables do not match in type.
    /// </summary>
    public class TypeMismatchException : Exception
    {
        public TypeMismatchException() { }
        public TypeMismatchException(string message) : base(message) { }
    }
}
