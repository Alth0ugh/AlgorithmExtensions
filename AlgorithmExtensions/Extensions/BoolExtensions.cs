using System;
using System.Collections.Generic;
using System.Linq;
using System.Runtime.CompilerServices;
using System.Text;
using System.Threading.Tasks;

namespace AlgorithmExtensions.Extensions
{
    internal static class BoolExtensions
    {
        internal static int ToInt(this bool value)
        {
            if (value)
                return 1;
            return 0;
        }
    }
}
