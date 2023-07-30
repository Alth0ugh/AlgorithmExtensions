namespace AlgorithmExtensions.Exceptions
{
    /// <summary>
    /// Thrown when the property representing the data is not in expected type.
    /// </summary>
    public class PropertyTypeException : Exception
    {
        public PropertyTypeException() { }
        public PropertyTypeException(string message) : base(message) { }
    }
}
