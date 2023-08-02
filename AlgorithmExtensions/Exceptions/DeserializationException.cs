namespace AlgorithmExtensions.Exceptions
{
    [Serializable]
    internal class DeserializationException : Exception
    {
        public DeserializationException() { }
        public DeserializationException(string? message) : base(message) { }
    }
}