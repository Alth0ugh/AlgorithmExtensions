namespace AlgorithmExtensions.Hyperalgorithms.ParameterProviders
{
    /// <summary>
    /// Represents an instance of parameter with assigned value.
    /// </summary>
    public class ParameterInstance
    {
        /// <summary>
        /// Name of the parameter.
        /// </summary>
        public string Name { get; set; }
        /// <summary>
        /// Value of the parameter.
        /// </summary>
        public object Value { get; set; }
        public ParameterInstance(string name, object value)
        {
            Name = name;
            Value = value;
        }
    }
}
