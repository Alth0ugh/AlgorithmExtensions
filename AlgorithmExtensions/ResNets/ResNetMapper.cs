using AlgorithmExtensions.Exceptions;
using AlgorithmExtensions.Hyperalgorithms;
using Microsoft.ML;
using Microsoft.ML.Data;
using System.Data;

namespace AlgorithmExtensions.ResNets
{
    /// <summary>
    /// Row to row mapper for ResNet.
    /// </summary>
    public class ResNetMapper : ResNetBase, IRowToRowMapper
    {
        public DataViewSchema InputSchema => throw new NotImplementedException();

        public DataViewSchema OutputSchema => throw new NotImplementedException();
        
        private readonly ResNetTransformer _model;
        private readonly Options _options;

        internal ResNetMapper(ResNetTransformer model, Options options)
        {
            _model = model;
            _options = options;
        }

        public IEnumerable<DataViewSchema.Column> GetDependencies(IEnumerable<DataViewSchema.Column> dependingColumns)
        {
            var featureColumn = from column in dependingColumns
                                where column.Name == _options.FeatureColumnName
                                select column;
            
            if (featureColumn.Count() != 1)
            {
                throw new MissingColumnException($"Column with name {_options.FeatureColumnName} is missing from the input.");
            }

            
            return new[] { featureColumn.ElementAt(0) };
        }

        public DataViewRow GetRow(DataViewRow input, IEnumerable<DataViewSchema.Column> activeColumns)
        {
            var featureColumn = from column in activeColumns
                                where column.Name == _options.FeatureColumnName
                                select column;

            if (featureColumn.Count() != 1)
            {
                throw new MissingColumnException($"Column with name {_options.FeatureColumnName} is missing from the input.");
            }
            
            var featureGetter = input.GetGetter<MLImage>(featureColumn.ElementAt(0));

            MLImage image = null;

            featureGetter(ref image);

            var result = _model.Transform(image);
            var cursor = result.GetRowCursor(result.Schema);
            cursor.MoveNext();
            return cursor;
        }
    }
}
