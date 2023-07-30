using Microsoft.ML;
using Microsoft.ML.Data;
using Tensorflow.Keras.Engine;

namespace AlgorithmExtensions.ResNets
{
    public class ResNetTransformer : ITransformer
    {
        private IModel _model;

        internal ResNetTransformer(IModel model)
        {
            _model = model;     
        }

        public bool IsRowToRowMapper => throw new NotImplementedException();

        public DataViewSchema GetOutputSchema(DataViewSchema inputSchema)
        {
            throw new NotImplementedException();
        }

        public IRowToRowMapper GetRowToRowMapper(DataViewSchema inputSchema)
        {
            throw new NotImplementedException();
        }

        public void Save(ModelSaveContext ctx)
        {
            throw new NotImplementedException();
        }

        public IDataView Transform(IDataView input)
        {
            throw new NotImplementedException();
        }
    }
}
