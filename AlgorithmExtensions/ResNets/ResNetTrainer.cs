using Microsoft.ML;
using static Tensorflow.Binding;
using Tensorflow;
using Tensorflow.Keras.Engine;
using AlgorithmExtensions.Extensions;
using AlgorithmExtensions.ResNets.Blocks;
using Microsoft.ML.Data;
using Tensorflow.NumPy;
using AlgorithmExtensions.Exceptions;

namespace AlgorithmExtensions.ResNets
{
    /// <summary>
    /// Trainer for ResNet.
    /// </summary>
    public class ResNetTrainer : IEstimator<ResNetTransformer>
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
            model.compile("adam", "sparse_categorical_crossentropy", new string[] { "accuracy" });
            return model;
        }

        /// <summary>
        /// Fits the model.
        /// </summary>
        /// <param name="input">Data to fit the model on.</param>
        /// <returns></returns>
        public ResNetTransformer Fit(IDataView input)
        {
            var featureColumn = input.Schema[_options.FeatureColumnName];
            var labelColumn = input.Schema[_options.LabelColumnName];

            var labelCursor = input.GetRowCursor(new[] { labelColumn });
            var labelGetter = labelCursor.GetGetter<uint>(labelColumn);
            var y = GetLabels(labelCursor, labelGetter);

            var featureCursor = input.GetRowCursor(new[] { featureColumn });
            var imageDataGetter = featureCursor.GetGetter<MLImage>(featureColumn);
            var x = GetInputData(featureCursor, imageDataGetter) / 255.0f;

            _model.fit(x, y, batch_size: _options.BatchSize, epochs: _options.Epochs);

            return new ResNetTransformer(_model);
        }

        /// <summary>
        /// Loads input images into NDArray.
        /// </summary>
        /// <param name="cursor">Cursor pointing to input data.</param>
        /// <param name="imageDataGetter">Getter for image data.</param>
        /// <returns>NDArray containg image data.</returns>
        private NDArray GetInputData(DataViewRowCursor cursor, ValueGetter<MLImage> imageDataGetter)
        {
            var list = new List<byte[,,]>();
            var width = 0;
            var height = 0;
            while (cursor.MoveNext())
            {
                MLImage imageValue = default;
                imageDataGetter(ref imageValue);
                width = imageValue.Width;
                height = imageValue.Height;
                var pixels = GetPixelsFromImage(imageValue).ToByteArray();
                list.Add(pixels);
            }

            var resultArray = new NDArray(new Shape(list.Count, height, width, 3), TF_DataType.TF_UINT8);
            
            for (int i = 0; i < list.Count; i++)
            {
                resultArray[i] = list[i];
            }

            return resultArray;
        }

        /// <summary>
        /// Converts image into pixel array,
        /// </summary>
        /// <param name="image">Input image.</param>
        /// <returns>2D pixel array.</returns>
        private Pixel[,] GetPixelsFromImage(MLImage image)
        {
            var result = new Pixel[image.Height,image.Width];
            int i = 0;
            int j = 0;
            int k = 0;

            while (k < image.Width * image.Height * 4)
            {
                result[i,j] = GetPixelFromImage(image, k);
                if ((j + 1) % image.Width == 0)
                {
                    i++;
                }
                j = (j + 1) % image.Width;
                k += 4;
            }

            return result;
        }

        /// <summary>
        /// Returns pixel from byte array representing an image.
        /// </summary>
        /// <param name="image">Input image.</param>
        /// <param name="index">Index in the byte array pointing to the first byte of a pixel.</param>
        /// <returns>Pixel from the image.</returns>
        /// <exception cref="UnknownPixelFormatException">Thrown when pixel format of input data is not recognized.</exception>
        private Pixel GetPixelFromImage(MLImage image, int index)
        {
            switch (image.PixelFormat)
            {
                case MLPixelFormat.Rgba32:
                    {
                        var r = image.Pixels[index];
                        var g = image.Pixels[index + 1];
                        var b = image.Pixels[index + 2];
                        var a = image.Pixels[index + 3];
                        return new Pixel(r, g, b, a);
                    }
                case MLPixelFormat.Bgra32:
                    {
                        var b = image.Pixels[index];
                        var g = image.Pixels[index + 1];
                        var r = image.Pixels[index + 2];
                        var a = image.Pixels[index + 3];
                        return new Pixel(r, g, b, a);
                    }
                default:
                    throw new UnknownPixelFormatException("Pixel format of input images is unknown. Use either RGBA or BGRA."); // TODO
            }
        }

        /// <summary>
        /// Gets labels from input data.
        /// </summary>
        /// <param name="cursor">Cursor pointing to input data.</param>
        /// <param name="labelGetter">Getter for label.</param>
        /// <returns>1D NDArray containing labels.</returns>
        private NDArray GetLabels(DataViewRowCursor cursor, ValueGetter<uint> labelGetter)
        {
            uint label = default;

            var labels = new List<uint>();

            while (cursor.MoveNext())
            {
                labelGetter(ref label);
                labels.Add(label);
            }

            return new NDArray(labels.ToArray());
        }

        /// <summary>
        /// Returns the schema of ouput with regards to the input data.
        /// </summary>
        /// <param name="inputSchema">Schema of the input data.</param>
        /// <returns>Output schema.</returns>
        public SchemaShape GetOutputSchema(SchemaShape inputSchema)
        {
            throw new NotImplementedException();
        }
    }
}