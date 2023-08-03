using AlgorithmExtensions.Hyperalgorithms;
using Microsoft.ML;
using Microsoft.ML.Data;
using Tensorflow;
using Tensorflow.Keras.Engine;
using Tensorflow.NumPy;
using AlgorithmExtensions.Extensions;
using static Tensorflow.KerasApi;
using static Tensorflow.Binding;
using AlgorithmExtensions.Exceptions;
using Newtonsoft.Json;
using SkiaSharp;
using Tensorflow.Operations.Initializers;

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

        internal ResNetTransformer(IModel model, Options options, DataViewSchema inputSchema)
        {
            _model = model;
            _options = options;
            _mlContext = new MLContext();
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
            var featureColumn = from column in inputSchema
                      where column.Name == _options.FeatureColumnName
                      select column;
            if (featureColumn.Count() == 0)
            {
                throw new MissingColumnException($"{nameof(inputSchema)} does not containt column named {_options.FeatureColumnName}");
            }
            return new ResNetMapper(this, _options);
        }

        public void Save(ModelSaveContext ctx)
        {
            throw new NotImplementedException();
            Save(Directory.GetCurrentDirectory(), "model");
        }

        /// <summary>
        /// Saves model to a given path.
        /// </summary>
        /// <param name="path">Path to save the model.</param>
        /// <exception cref="IOException">Thrown when the model is saved unsucessfully.</exception>
        public void Save(string path, string modelName)
        {
            throw new NotImplementedException();
            try
            {
                _model.save(Path.Combine(path, modelName));
                var serializer = new JsonSerializer();
                var modelStream = new StreamWriter(Path.Combine(path, modelName + ".model"));
                var schemaStream = new StreamWriter(Path.Combine(path, modelName + ".input"));
                var optionsStream = new StreamWriter(Path.Combine(path, modelName + ".options"));

                serializer.Serialize(modelStream, _model);
                serializer.Serialize(schemaStream, InputSchema);
                serializer.Serialize(optionsStream, _options);

                schemaStream.Dispose();
                optionsStream.Dispose();
            }
            catch (Exception ex)
            {
                throw new IOException("Model could not be saved in the given path. See inner exception.", ex);
            }
        }

        /// <summary>
        /// Load model.
        /// </summary>
        /// <param name="mlContext">Machine learning context.</param>
        /// <param name="path">Path to folder where the model files are stored.</param>
        /// <param name="modelName">Name of the model without the extension.</param>
        /// <returns>Loaded model.</returns>
        /// <exception cref="IOException">Thrown when any of the model files cannot be read or is not present.</exception>
        /// <exception cref="DeserializationException">Thrown when any of the model files cannot be deserialized.</exception>
        public static ResNetTransformer Load(string path, string modelName)
        {
            throw new NotImplementedException();
            IModel? kerasModel = default;
            DataViewSchema? schema = default;
            Options? options = default;

            try
            {
                var loadOptions = new LoadOptions();
                loadOptions.experimental_variable_policy = VariablePolicy.SAVE_VARIABLE_DEVICES;
                kerasModel = tf.keras.models.load_model(Path.Combine(path, modelName), true, loadOptions);
            }
            catch (Exception ex)
            {
                throw new IOException("Error while loading model with .keras extension. See inner exception", ex);
            }

            try
            {
                var schemaJson = File.ReadAllText(Path.Combine(path, modelName + ".input"));
                //schema = JsonSerializer.Deserialize<DataViewSchema>(schemaJson);
            }
            catch (Exception ex)
            {
                throw new IOException("Schema with .input extension could not be loaded or deserialized. See inner exception.", ex);
            }

            try
            {
                var optionsJson = File.ReadAllText(Path.Combine(path, modelName + ".options"));
                //ptions = JsonSerializer.Deserialize<Options>(optionsJson);
            }
            catch (Exception ex)
            {
                throw new IOException("Options with .options extension could not be loaded or deserialized. See inner exception.", ex);
            }

            if (options == null || schema == null)
            {
                throw new DeserializationException(options == null ? "Options could not be deserialized" : "Schema could not be deserialized");
            }

            return new ResNetTransformer(kerasModel, options, schema);
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
            var x = new NDArray(new Shape(1, image.Height, image.Width, 3), TF_DataType.TF_UINT8);
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
