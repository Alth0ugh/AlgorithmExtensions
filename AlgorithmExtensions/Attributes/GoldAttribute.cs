namespace AlgorithmExtensions.Attributes
{
    /// <summary>
    /// Decorates property representing gold labels of data.
    /// </summary>
    [AttributeUsage(AttributeTargets.Property, AllowMultiple = false, Inherited = false)]
    public class GoldAttribute : Attribute
    {
    }
}
