﻿namespace AlgorithmExtensions.Exceptions
{
    /// <summary>
    /// Thrown when property name is not found in option object for model.
    /// </summary>
    public class IncorrectOptionParameterException : Exception
    {
        public IncorrectOptionParameterException() { }
        public IncorrectOptionParameterException(string message) : base(message) { }
    }
}
