namespace AlgorithmExtensions.Attributes
{
    /// <summary>
    /// Decorates property representing predicted labels of data.
    /// </summary>
    [AttributeUsage(AttributeTargets.Property, AllowMultiple = false, Inherited = true)]
    public class PredictionAttribute : Attribute
    {

    }
}
