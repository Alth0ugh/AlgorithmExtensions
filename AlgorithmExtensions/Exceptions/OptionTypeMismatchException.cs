namespace AlgorithmExtensions.Exceptions
{
    /// <summary>
    /// Thrown when type of parameter supplied by user does not match the type of parameter in option object.
    /// </summary>
    public class OptionTypeMismatchException : TypeMismatchException
    {
        public Type? ModelOptionType { get; init; }
        public Type? SuppliedType { get; init; }
        public OptionTypeMismatchException() { }
        public OptionTypeMismatchException(string message) : base(message) { }
        public OptionTypeMismatchException(string message, Type modelOptionType, Type suppliedType) : base(message)
        {
            ModelOptionType = modelOptionType;
            SuppliedType = suppliedType;
        }
    }
}
