﻿using AlgorithmExtensions.Hyperalgorithms;
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
            var labelColumns = from column in inputSchema
                               where column.Name == _options.LabelColumnName
                               select column;

            var builder = new DataViewSchema.Builder();

            if (labelColumns.Count() > 0)
            {
                builder.AddColumn(nameof(ModelPredictionWithGold.Prediction), NumberDataViewType.Single);
                builder.AddColumn(nameof(ModelPredictionWithGold.Gold), NumberDataViewType.Single);
            }
            else
            {
                builder.AddColumn(nameof(ModelPredictionWithoutGold.Prediction), NumberDataViewType.Single);
            }

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
                _model.save(Path.Combine(path, modelName));
            }
            catch (Exception ex)
            {
                throw new IOException("Model could not be saved in the given path. See inner exception.", ex);
            }
        }

        public IDataView Transform(IDataView input)
        {
            var labelColumns = from column in input.Schema
                               where column.Name == _options.LabelColumnName
                               select column;
            if (labelColumns.Count() > 0)
            {
                return TransformWithGold(input);
            }
            else
            {
                return TransformWithoutGold(input);
            }
            
        }

        private IDataView TransformWithGold(IDataView input)
        {
            var featureColumn = input.Schema[_options.FeatureColumnName];
            var labelColumn = input.Schema[_options.LabelColumnName];

            var labelCursor = input.GetRowCursor(new[] { labelColumn });
            var labelGetter = labelCursor.GetGetter<uint>(labelColumn);
            var y = GetLabels(labelCursor, labelGetter, _options.Classes);

            var featureCursor = input.GetRowCursor(new[] { featureColumn });
            var imageDataGetter = featureCursor.GetGetter<MLImage>(featureColumn);
            var x = GetInputData(featureCursor, imageDataGetter, _options.Height, _options.Width) / 255.0f;

            var modelPrediction = _model.predict(x);
            var predictions = GetClassesFromOneHot(modelPrediction);
            var golds = GetClassesFromOneHot(y);

            var result = new List<ModelPredictionWithGold>();

            for (int i = 0; i < predictions.Length; i++)
            {
                result.Add(new ModelPredictionWithGold() { Gold = golds[i], Prediction = predictions[i] });
            }

            return _mlContext.Data.LoadFromEnumerable(result);
        }

        private IDataView TransformWithoutGold(IDataView input)
        {
            var featureColumn = input.Schema[_options.FeatureColumnName];

            var featureCursor = input.GetRowCursor(new[] { featureColumn });
            var imageDataGetter = featureCursor.GetGetter<MLImage>(featureColumn);
            var x = GetInputData(featureCursor, imageDataGetter, _options.Height, _options.Width) / 255.0f;

            var modelPrediction = _model.predict(x);
            var predictions = GetClassesFromOneHot(modelPrediction);

            var result = predictions.Select(x => new ModelPredictionWithoutGold() { Prediction = x });

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
            var predictions = GetClassesFromOneHot(_model.predict(x));
            var result = predictions.Select(x => new ModelPredictionWithoutGold() { Prediction = x });

            return _mlContext.Data.LoadFromEnumerable(result);
        }

        private uint[] GetClassesFromOneHot(Tensors tensors)
        {
            Tensor[] ten = tensors;
            var count = tensors.shape[0];
            var predictions = new uint[count];
            for (int i = 0; i < count; i++)
            {
                var prediction = np.argmax(ten[0][i].numpy());
                predictions[i] = prediction;
            }

            return predictions;
        }
    }
}
