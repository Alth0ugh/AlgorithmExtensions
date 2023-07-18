using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace AlgorithmExtensions.Exceptions
{
    public class OptionTypeMismatchException : TypeMismatchException
    {
        public Type ModelOptionType { get; init; }
        public Type SuppliedType { get; init; }
        public OptionTypeMismatchException() : base() { }
        public OptionTypeMismatchException(string message) : base(message) { }
        public OptionTypeMismatchException(string message, Type modelOptionType, Type suppliedType) : base(message)
        {
            ModelOptionType = modelOptionType;
            SuppliedType = suppliedType;
        }
    }
}
