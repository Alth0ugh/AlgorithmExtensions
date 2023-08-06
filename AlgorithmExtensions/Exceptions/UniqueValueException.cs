namespace AlgorithmExtensions.Exceptions
{
    /// <summary>
    /// Thrown when a collection does not contain unique values.
    /// </summary>
    public class UniqueValueException : Exception
    {
        public UniqueValueException() { }
        public UniqueValueException(string message) : base(message) { }
    }
}
