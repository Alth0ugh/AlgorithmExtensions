using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace AlgorithmExtensions.Exceptions
{
    public class MissingAttributeException : Exception
    {
        public MissingAttributeException() : base() { }
        public MissingAttributeException(string message) : base(message) { }
    }
}
