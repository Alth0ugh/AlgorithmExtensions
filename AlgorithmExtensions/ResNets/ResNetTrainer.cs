using Microsoft.ML;
using static Tensorflow.Binding;
using Tensorflow;
using Tensorflow.Keras.Engine;
using AlgorithmExtensions.Extensions;
using AlgorithmExtensions.ResNets.Blocks;
using Microsoft.ML.Data;
using Tensorflow.NumPy;
using static Microsoft.ML.SchemaShape.Column;
using AlgorithmExtensions.Exceptions;

namespace AlgorithmExtensions.ResNets
{
    /// <summary>
    /// Trainer for ResNet.
    /// </summary>
    public class ResNetTrainer : ResNetBase, IEstimator<ResNetTransformer>
    {
        private Options _options;
        private IModel _model;

        private const string _checkInputColumnsColumnMissingError = "Column named {0} is missing from the input";
        private const string _checkInputColumnsTypeMismatchError = "Expected type {0} for {1} but got {2}";

        public ResNetTrainer(Options options)
        {
            _options = options;
            _model = GenerateModel(options.Architecture);
        }

        /// <summary>
        /// Fits the model.
        /// </summary>
        /// <param name="input">Data to fit the model on.</param>
        /// <returns>Trained transformer.</returns>
        /// <exception cref="MissingColumnException">Thrown when a column is missing from the input data.</exception>
        /// <exception cref="TypeMismatchException">Thrown when the expected data type of a column is different from expected data type.</exception>
        /// <exception cref="IncorrectDimensionsException">Thrown when the input dimensions are below 32 pixels in any dimension.</exception>
        public ResNetTransformer Fit(IDataView input)
        {
            CheckInputDimensions();
            CheckInputColumns(input);

            var featureColumn = input.Schema[_options.FeatureColumnName];
            var labelColumn = input.Schema[_options.LabelColumnName];

            var labelCursor = input.GetRowCursor(new[] { labelColumn });
            var labelGetter = labelCursor.GetGetter<uint>(labelColumn);
            var y = GetLabels(labelCursor, labelGetter, _options.Classes);

            var featureCursor = input.GetRowCursor(new[] { featureColumn });
            var imageDataGetter = featureCursor.GetGetter<MLImage>(featureColumn);
            var x = GetInputData(featureCursor, imageDataGetter, _options.Height, _options.Width) / 255.0f;

            _model.fit(x, y, batch_size: _options.BatchSize, epochs: _options.Epochs);

            labelCursor.Dispose();
            featureCursor.Dispose();

            return new ResNetTransformer(_model, _options, input.Schema);
        }

        /// <summary>
        /// Returns the schema of ouput with regards to the input data.
        /// </summary>
        /// <param name="inputSchema">Schema of the input data.</param>
        /// <exception cref="DependencyException">Thrown when the output schema cannot be created due to some error in dependent library.</exception>
        /// <returns>Output schema.</returns>
        public SchemaShape GetOutputSchema(SchemaShape inputSchema)
        {
            try
            {
                var constructor = typeof(SchemaShape.Column).GetConstructor(System.Reflection.BindingFlags.NonPublic | System.Reflection.BindingFlags.Instance, new Type[] { typeof(string), typeof(VectorKind), typeof(DataViewType), typeof(bool), typeof(SchemaShape) })!;
                var predictionColumn = (SchemaShape.Column)constructor.Invoke(new object[] { nameof(ModelPrediction.Prediction), VectorKind.Scalar, NumberDataViewType.Single, false, null });
                return new SchemaShape(new[] { predictionColumn });
            }
            catch (Exception ex)
            {
                throw new DependencyException("Could not create output schema due to dependency error.", ex);
            }
        }

        private void CheckInputDimensions()
        {
            if (_options.Width < 32 || _options.Height < 32)
            {
                throw new IncorrectDimensionsException($"{nameof(Options.Width)} and {nameof(Options.Height)} must be at least 32.");
            }
        }

        /// <summary>
        /// Generates ResNet with a given architecture.
        /// </summary>
        /// <param name="architecture">Architecture of ResNet.</param>
        /// <returns>ResNet model.</returns>
        private IModel GenerateModel(ResNetArchitecture architecture)
        {
            var input = tf.keras.layers.Input(new Shape(_options.Height, _options.Width, 3));
            var conv1 = tf.keras.layers.Conv2D(64,
                new Shape(7, 7),
                new Shape(2, 2),
                padding: "same",
                bias_initializer: "zeros").Apply(input);
            var pool = tf.keras.layers.MaxPooling2D(new Shape(3, 3), new Shape(2, 2)).Apply(conv1);

            Tensors? residual = null;
            switch(architecture)
            {
                case ResNetArchitecture.ResNet50:
                    {
                        var block = new BottleneckBlock();
                        residual = pool.StackResidualBlocks(block, 64, new Shape(1, 1), 3)
                                .StackResidualBlocks(block, 128, new Shape(2, 2), 4)
                                .StackResidualBlocks(block, 256, new Shape(2, 2), 6)
                                .StackResidualBlocks(block, 512, new Shape(2, 2), 3);
                        break;
                    }
                case ResNetArchitecture.ResNet101:
                    {
                        var block = new BottleneckBlock();
                        residual = pool.StackResidualBlocks(block, 64, new Shape(1, 1), 3)
                                .StackResidualBlocks(block, 128, new Shape(2, 2), 4)
                                .StackResidualBlocks(block, 256, new Shape(2, 2), 23)
                                .StackResidualBlocks(block, 512, new Shape(2, 2), 3);
                        break;
                    }
                case ResNetArchitecture.ResNet152:
                    {
                        var block = new BottleneckBlock();
                        residual = pool.StackResidualBlocks(block, 64, new Shape(1, 1), 3)
                                .StackResidualBlocks(block, 128, new Shape(2, 2), 8)
                                .StackResidualBlocks(block, 256, new Shape(2, 2), 36)
                                .StackResidualBlocks(block, 512, new Shape(2, 2), 3);
                        break;
                    }
                case ResNetArchitecture.ResNet18:
                    {
                        var block = new BasicBlock();
                        residual = pool.StackResidualBlocks(block, 64, new Shape(1, 1), 2)
                            .StackResidualBlocks(block, 128, new Shape(2, 2), 2)
                            .StackResidualBlocks(block, 256, new Shape(2, 2), 2)
                            .StackResidualBlocks(block, 512, new Shape(2, 2), 2);
                        break;
                    }
                case ResNetArchitecture.ResNet34:
                    {
                        var block = new BasicBlock();
                        residual = pool.StackResidualBlocks(block, 64, new Shape(1, 1), 3)
                            .StackResidualBlocks(block, 128, new Shape(2, 2), 4)
                            .StackResidualBlocks(block, 256, new Shape(2, 2), 6)
                            .StackResidualBlocks(block, 512, new Shape(2, 2), 3);
                        break;
                    }
                default:
                    throw new UnknownArchitectureException("The given ResNet architecture is unknown.");
            }

            var maxPool = tf.keras.layers.GlobalAveragePooling2D().Apply(residual);
            var flatten = tf.keras.layers.Flatten().Apply(maxPool);
            var softmax = tf.keras.layers.Dense(_options.Classes, activation: "softmax").Apply(flatten);

            var model = tf.keras.Model(input, softmax);
            model.compile(tf.keras.optimizers.Adam(_options.LearningRate), tf.keras.losses.BinaryCrossentropy(), new string[] { "accuracy" });
            return model;
        }

        /// <summary>
        /// Checks if the input columns are present and in correct data type.
        /// </summary>
        /// <param name="input">Input data.</param>
        /// <exception cref="MissingColumnException">Thrown when a column is missing from the input data.</exception>
        /// <exception cref="TypeMismatchException">Thrown when the expected data type of a column is different from expected data type.</exception>
        private void CheckInputColumns(IDataView input)
        {
            var featureColumn = from column in input.Schema
                                      where column.Name == _options.FeatureColumnName
                                      select column;
            var labelColumn = from column in input.Schema
                                    where column.Name == _options.LabelColumnName
                                    select column;

            var featureColumnCount = featureColumn.Count();
            var labelColumnCount = labelColumn.Count();
            if (featureColumnCount == 0 || labelColumnCount == 0)
            {
                throw new MissingColumnException(featureColumnCount == 0 ? string.Format(_checkInputColumnsColumnMissingError, _options.FeatureColumnName) :
                    string.Format(_checkInputColumnsColumnMissingError, _options.LabelColumnName));
            }

            var featureColumnInstance = featureColumn.Single();
            if (featureColumnInstance.Type.RawType != typeof(MLImage))
            {
                throw new TypeMismatchException(string.Format(_checkInputColumnsTypeMismatchError, typeof(MLImage), _options.FeatureColumnName, featureColumnInstance.Type.RawType));
            }

            var labelColumnInstance = labelColumn.Single();
            if (labelColumnInstance.Type.RawType != typeof(uint))
            {
                throw new TypeMismatchException(string.Format(_checkInputColumnsTypeMismatchError, typeof(MLImage), _options.LabelColumnName, labelColumnInstance.Type.RawType));
            }
        }
    }
}