namespace AlgorithmExtensions.Exceptions
{
    public class TypeMismatchException : Exception
    {
        public TypeMismatchException() : base() { }
        public TypeMismatchException(string message) : base(message) { }
    }
}
