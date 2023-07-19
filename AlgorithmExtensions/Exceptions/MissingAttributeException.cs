namespace AlgorithmExtensions.Exceptions
{
    public class MissingAttributeException : Exception
    {
        public MissingAttributeException() : base() { }
        public MissingAttributeException(string message) : base(message) { }
    }
}
