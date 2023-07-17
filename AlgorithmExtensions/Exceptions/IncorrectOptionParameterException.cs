using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace AlgorithmExtensions.Exceptions
{
    public class IncorrectOptionParameterException : Exception
    {
        public IncorrectOptionParameterException() : base() { }
        public IncorrectOptionParameterException(string message) : base(message) { }
    }
}
