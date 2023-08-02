﻿using Microsoft.ML;
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

        public ResNetTrainer(Options options)
        {
            _options = options;
            _model = GenerateModel(options.Architecture);
        }

        /// <summary>
        /// Generates ResNet with a given architecture.
        /// </summary>
        /// <param name="architecture">Architecture of ResNet.</param>
        /// <returns>ResNet model.</returns>
        private IModel GenerateModel(ResNetArchitecture architecture)
        {
            var input = tf.keras.layers.Input(new Shape(32, 32, 3));
            var conv1 = tf.keras.layers.Conv2D(64,
                new Shape(7, 7),
                new Shape(2, 2),
                padding: "same",
                bias_initializer: "zeros").Apply(input);
            var pool = tf.keras.layers.MaxPooling2D(new Shape(3, 3), new Shape(2, 2)).Apply(conv1);

            Tensors residual = null;
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
            //var dense = tf.keras.layers.Dense(1000).Apply(flatten);
            var softmax = tf.keras.layers.Dense(_options.Classes, activation: "softmax").Apply(flatten);

            var model = tf.keras.Model(input, softmax);
            model.compile(tf.keras.optimizers.Adam(0.0001f), tf.keras.losses.BinaryCrossentropy(), new string[] { "accuracy" });
            //model.compile(optimizer: tf.keras.optimizers.RMSprop(1e-3f),
            //    loss: tf.keras.losses.SparseCategoricalCrossentropy(from_logits: true),
            //    metrics: new[] { "acc" });
            return model;
        }

        /// <summary>
        /// Fits the model.
        /// </summary>
        /// <param name="input">Data to fit the model on.</param>
        /// <returns>Trained transformer.</returns>
        public ResNetTransformer Fit(IDataView input)
        {
            var featureColumn = input.Schema[_options.FeatureColumnName];
            var labelColumn = input.Schema[_options.LabelColumnName];

            var labelCursor = input.GetRowCursor(new[] { labelColumn });
            var labelGetter = labelCursor.GetGetter<uint>(labelColumn);
            var y = GetLabels(labelCursor, labelGetter, _options.Classes);

            var featureCursor = input.GetRowCursor(new[] { featureColumn });
            var imageDataGetter = featureCursor.GetGetter<MLImage>(featureColumn);
            var x = GetInputData(featureCursor, imageDataGetter) / 255.0f;

            _model.fit(x, y, batch_size: _options.BatchSize, epochs: _options.Epochs);

            labelCursor.Dispose();
            featureCursor.Dispose();

            return new ResNetTransformer(_model, _options, input.Schema);
        }

        public ResNetTransformer Fit(NDArray x, NDArray y)
        {
            _model.fit(x, y, batch_size: _options.BatchSize, epochs: _options.Epochs);
            return new ResNetTransformer(_model, _options, null);
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
                var constructor = typeof(SchemaShape.Column).GetConstructor(System.Reflection.BindingFlags.NonPublic | System.Reflection.BindingFlags.Instance, new Type[] { typeof(string), typeof(VectorKind), typeof(DataViewType), typeof(bool), typeof(SchemaShape) });
                var predictionColumn = (SchemaShape.Column)constructor.Invoke(new object[] { "Probabilities", VectorKind.Scalar, NumberDataViewType.Single, false, null });
                return new SchemaShape(new[] { predictionColumn });
            }
            catch (Exception ex)
            {
                throw new DependencyException("Could not create output schema due to dependency error.", ex);
            }
        }
    }
}