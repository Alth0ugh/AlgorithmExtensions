namespace AlgorithmExtensions.Exceptions
{
    /// <summary>
    /// Thrown when creational delegate for PipelineTemplate is not in correct state (function taking 1 argument of type TrainerInputBase).
    /// </summary>
    public class IncorrectCreationalDelegateException : Exception
    {
        public IncorrectCreationalDelegateException() { }
        public IncorrectCreationalDelegateException(string message) : base(message) { }
    }
}
