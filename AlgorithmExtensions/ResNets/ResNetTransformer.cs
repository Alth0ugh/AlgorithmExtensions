using AlgorithmExtensions.Hyperalgorithms;
using Microsoft.ML;
using Microsoft.ML.Data;
using System.Security;
using Tensorflow;
using System.Linq;
using Tensorflow.Keras.Engine;
using Tensorflow.NumPy;

namespace AlgorithmExtensions.ResNets
{
    public class ResNetTransformer : ResNetBase, ITransformer
    {
        private IModel _model;
        private Options _options;
        private MLContext _mlContext;

        internal ResNetTransformer(IModel model, Options options, MLContext mlContext)
        {
            _model = model;
            _options = options;
            _mlContext = mlContext;
        }

        public bool IsRowToRowMapper => throw new NotImplementedException();

        public DataViewSchema GetOutputSchema(DataViewSchema inputSchema)
        {
            if (inputSchema == null)
            {
                throw new ArgumentException($"Parameter {nameof(inputSchema)} cannot be null");
            }
            var builder = new DataViewSchema.Builder();
            foreach (var column in inputSchema)
            {
                builder.AddColumn(column.Name, column.Type, column.Annotations);
            }

            builder.AddColumn(_options.PredictedLabelColumnName, NumberDataViewType.Single);
            return builder.ToSchema();
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
            var featureColumn = input.Schema[_options.FeatureColumnName];
            var labelColumn = input.Schema[_options.LabelColumnName];

            var featureCursor = input.GetRowCursor(new[] { featureColumn });
            var imageDataGetter = featureCursor.GetGetter<MLImage>(featureColumn);
            var x = GetInputData(featureCursor, imageDataGetter) / 255.0f;

            var modelPrediction = _model.predict(x);
            var predictions = GetPredictions(modelPrediction);

            var result = predictions.Select(x => new ModelPrediction() { Prediction = x });

            return _mlContext.Data.LoadFromEnumerable(result);
        }

        public IDataView Transform(NDArray x)
        {
            
            var modelPrediction = _model.predict(x);
            var predictions = GetPredictions(modelPrediction);

            var result = predictions.Select(x => new ModelPrediction() { Prediction = x });

            return _mlContext.Data.LoadFromEnumerable(result);
        }

        private uint[] GetPredictions(Tensors tensors)
        {
            var count = tensors.shape[0];
            var predictions = new uint[count];
            for (int i = 0; i < count; i++)
            {
                var prediction = np.argmax(tensors[0][i].numpy());
                predictions[i] = prediction;
            }

            return predictions;
        }
    }
}
