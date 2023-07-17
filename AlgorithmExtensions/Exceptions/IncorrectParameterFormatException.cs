using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace AlgorithmExtensions.Exceptions
{
    public class IncorrectParameterFormatException : Exception
    {
        public IncorrectParameterFormatException() : base() { }
        public IncorrectParameterFormatException(string message) : base(message) { }
    }
}
