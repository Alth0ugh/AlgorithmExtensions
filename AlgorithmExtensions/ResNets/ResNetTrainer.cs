using Microsoft.ML;
using static Tensorflow.Binding;
using Tensorflow;
using Tensorflow.Keras.Engine;
using AlgorithmExtensions.Extensions;
using AlgorithmExtensions.ResNets.Blocks;

namespace AlgorithmExtensions.ResNets
{
    public class ResNetTrainer : IEstimator<ImageClassificationModelParameters>
    {
        private Options _options;
        public IModel _model;
        public ResNetTrainer(Options options)
        {
            _options = options;
            _model = GenerateModel(options.Architecture);
        }

        private IModel GenerateModel(ResNetArchitecture architecture)
        {
            var input = tf.keras.layers.Input(new Shape(224, 224, 3));
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
            }

            var maxPool = tf.keras.layers.GlobalAveragePooling2D().Apply(residual);
            var flatten = tf.keras.layers.Flatten().Apply(maxPool);
            var dense = tf.keras.layers.Dense(1000).Apply(flatten);
            var softmax = tf.keras.layers.Dense(_options.Classes).Apply(dense);

            var model = tf.keras.Model(input, softmax);

            return model;
        }

        public ImageClassificationModelParameters Fit(IDataView input)
        {
            throw new NotImplementedException();
        }

        public SchemaShape GetOutputSchema(SchemaShape inputSchema)
        {
            throw new NotImplementedException();
        }
    }
}