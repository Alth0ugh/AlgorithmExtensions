using AlgorithmExtensions.Hyperalgorithms;
using Microsoft.ML;
using Microsoft.ML.Data;
using Tensorflow;
using Tensorflow.Keras.Engine;
using Tensorflow.NumPy;
using AlgorithmExtensions.Extensions;
using System.Text.Json;
using static Tensorflow.Binding;
using static Tensorflow.KerasApi;
using RestSharp;
using AlgorithmExtensions.Exceptions;

namespace AlgorithmExtensions.ResNets
{
    /// <summary>
    /// Transformer based on trained ResNet.
    /// </summary>
    public class ResNetTransformer : ResNetBase, ITransformer
    {
        private IModel _model;
        private Options _options;
        private MLContext _mlContext;
        public DataViewSchema InputSchema { get; }

        internal ResNetTransformer(IModel model, Options options, MLContext mlContext, DataViewSchema inputSchema)
        {
            _model = model;
            _options = options;
            _mlContext = mlContext;
            InputSchema = inputSchema;
        }

        public bool IsRowToRowMapper => true;

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
            return new ResNetMapper(this, _options);
        }

        public void Save(ModelSaveContext ctx)
        {
            Save(Directory.GetCurrentDirectory(), "model");
        }

        /// <summary>
        /// Saves model to a given path.
        /// </summary>
        /// <param name="path">Path to save the model.</param>
        /// <exception cref="IOException">Thrown when the model is saved unsucessfully.</exception>
        public void Save(string path, string modelName)
        {
            try
            {
                _model.save(Path.Combine(path, modelName + ".keras"));
                var serializedSchema = JsonSerializer.Serialize(InputSchema);
                var serializedOptions = JsonSerializer.Serialize(_options);
                File.WriteAllText(Path.Combine(path, modelName + ".input"), serializedSchema);
                File.WriteAllText(Path.Combine(path, modelName + ".options"), serializedOptions);
            }
            catch (Exception ex)
            {
                throw new IOException("Model could not be saved in the given path. See inner exception.", ex);
            }
        }

        public static ResNetTransformer Load(MLContext mlContext, string path, string modelName)
        {
            IModel? kerasModel = default;
            DataViewSchema? schema = default;
            Options? options = default;

            try
            {
                kerasModel = keras.models.load_model(Path.Combine(path, modelName + ".keras"));
            }
            catch (Exception ex)
            {
                throw new IOException("Error while loading model with .keras extension. See inner exception", ex);
            }

            try
            {
                var schemaJson = File.ReadAllText(Path.Combine(path, modelName + ".input"));
                schema = JsonSerializer.Deserialize<DataViewSchema>(schemaJson);
            }
            catch (Exception ex)
            {
                throw new IOException("Schema with .input extension could not be loaded or deserialized. See inner exception.", ex);
            }

            try
            {
                var optionsJson = File.ReadAllText(Path.Combine(path, modelName + ".options"));
                options = JsonSerializer.Deserialize<Options>(optionsJson);
            }
            catch (Exception ex)
            {
                throw new IOException("Options with .options extension could not be loaded or deserialized. See inner exception.", ex);
            }

            if (options == null || schema == null)
            {
                throw new DeserializationException(options == null ? "Options could not be deserialized" : "Schema could not be deserialized");
            }

            return new ResNetTransformer(kerasModel, options, mlContext, schema);
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

        /// <summary>
        /// Take the data in, make transformations, output the data.
        /// </summary>
        /// <param name="x">NDArray containing the input data.</param>
        /// <returns>Transformed data.</returns>
        public IDataView Transform(NDArray x)
        {
            var modelPrediction = _model.predict(x);
            var predictions = GetPredictions(modelPrediction);

            var result = predictions.Select(x => new ModelPrediction() { Prediction = x });

            return _mlContext.Data.LoadFromEnumerable(result);
        }

        /// <summary>
        /// Take the data in, make transformations, output the data.
        /// </summary>
        /// <param name="image">Image that should be classified.</param>
        /// <returns>DataView containing the classification result.</returns>
        public IDataView Transform(MLImage image)
        {
            var pixels = GetPixelsFromImage(image).ToByteArray();
            var x = new NDArray(new Shape(1, image.Height, image.Width), TF_DataType.TF_UINT8);
            x[0] = pixels;
            x /= 255.0f;
            var predictions = GetPredictions(_model.predict(x));
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
