using AlgorithmExtensions.Attributes;
using AlgorithmExtensions.Exceptions;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Reflection;
using System.Text;
using System.Threading.Tasks;

namespace AlgorithmExtensions.Scoring
{
    public class BinaryScoringFunctionBase<Tout> : NumericScoringFunctionBase<Tout>
    {
        public void CheckIfPropertyTypeIsBool(PropertyInfo propertyInfo)
        {
            if (propertyInfo == null)
            {
                return;
            }

            if (propertyInfo.PropertyType != typeof(bool))
            {
                throw new PropertyTypeException($"Properties with attributes {typeof(GoldAttribute)} or {typeof(PredictionAttribute)} should be of type {typeof(bool)}");
            }
        }
    }
}
