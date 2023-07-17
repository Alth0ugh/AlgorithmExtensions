using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace AlgorithmExtensions.Exceptions
{
    public class ParameterConversionException : Exception
    {
        public ParameterConversionException() : base() { }
        public ParameterConversionException(string message) : base(message) { }
    }
}
