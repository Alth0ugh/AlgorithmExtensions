using AlgorithmExtensions.Attributes;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace AlgorithmExtensions.Tests
{
    public class GitHubIssueOutput : GihubIssue
    {
        public GitHubIssueOutput()
        {
            
        }
        [Gold]
        public uint Label { get; set; }
        [Prediction]
        public uint PredictedLabel { get; set; }
    }
}
