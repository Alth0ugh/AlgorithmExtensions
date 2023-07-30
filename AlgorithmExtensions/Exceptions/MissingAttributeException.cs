namespace AlgorithmExtensions.Exceptions
{
    /// <summary>
    /// Thrown when object representing data structure misses properties decorated with GoldAttribude and PredictionAttribute.
    /// </summary>
    public class MissingAttributeException : Exception
    {
        public MissingAttributeException() { }
        public MissingAttributeException(string message) : base(message) { }
    }
}
