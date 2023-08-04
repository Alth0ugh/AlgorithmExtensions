namespace AlgorithmExtensions.Exceptions
{
    /// <summary>
    /// Thrown when a column or multiple columns are missing from IDataView.
    /// </summary>
    public class MissingColumnException : Exception
    {
        public MissingColumnException() { }
        public MissingColumnException(string message) : base(message) { }
    }
}
