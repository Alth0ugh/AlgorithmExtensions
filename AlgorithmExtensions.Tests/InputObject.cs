using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace AlgorithmExtensions.Tests
{
    internal class InputObject
    {
        public string Features { get; set; }
        public bool Label { get; set; }

        public InputObject(bool label, string features)
        {
            Features = features;
            Label = label;
        }
    }
}
