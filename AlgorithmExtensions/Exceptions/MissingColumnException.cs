namespace AlgorithmExtensions.Exceptions
{

    public class MissingColumnException : Exception
    {
        public MissingColumnException() { }
        public MissingColumnException(string message) : base(message) { }
    }
}
