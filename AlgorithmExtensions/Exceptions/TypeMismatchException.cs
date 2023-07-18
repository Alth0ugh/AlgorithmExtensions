using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace AlgorithmExtensions.Exceptions
{
    public class TypeMismatchException : Exception
    {
        public TypeMismatchException() : base() { }
        public TypeMismatchException(string message) : base(message) { }
    }
}
