﻿using AlgorithmExtensions.Attributes;
using Microsoft.ML.Data;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace AlgorithmExtensions.Tests
{
    public class ModelOutput : ModelInput
    {
        [Prediction]
        [ColumnName("PredictedLabel")]
        public bool Prediction { get; set; }

        public float Score { get; set; }
    }
}