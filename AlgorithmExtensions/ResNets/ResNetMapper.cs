using AlgorithmExtensions.Exceptions;
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
        public DataViewSchema InputSchema => _model.InputSchema;

        public DataViewSchema OutputSchema => _model.GetOutputSchema(InputSchema);
        
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
            var dependencies = GetDependencies(input.Schema);
            var activeDependencyColumns = from column in activeColumns
                                          where dependencies.ElementAt(0).Name == column.Name
                                          select column;

            if (activeDependencyColumns.Count() != 1)
            {
                throw new MissingColumnException($"Column {dependencies.ElementAt(0)} is missing from {nameof(activeColumns)}");
            }

            var featureGetter = input.GetGetter<MLImage>(dependencies.ElementAt(0));

            MLImage image = null;

            featureGetter(ref image);

            var result = _model.Transform(image);
            var cursor = result.GetRowCursor(result.Schema);
            cursor.MoveNext();
            return cursor;
        }
    }
}
