using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace AlgorithmExtensions.Exceptions
{
    public class PropertyTypeException : Exception
    {
        public PropertyTypeException() : base() { }
        public PropertyTypeException(string message) : base(message) { }
    }
}
