using System.Runtime.Serialization;

namespace AlgorithmExtensions.ResNets
{
    [Serializable]
    internal class DeserializationException : Exception
    {
        public DeserializationException() { }
        public DeserializationException(string? message) : base(message) { }
    }
}