﻿using AlgorithmExtensions.Exceptions;
using Microsoft.ML.Data;
using Microsoft.ML;
using Tensorflow.NumPy;
using Tensorflow;
using AlgorithmExtensions.Extensions;

namespace AlgorithmExtensions.ResNets
{
    public class ResNetBase
    {
        /// <summary>
        /// Loads input images into NDArray.
        /// </summary>
        /// <param name="cursor">Cursor pointing to input data.</param>
        /// <param name="imageDataGetter">Getter for image data.</param>
        /// <returns>NDArray containg image data.</returns>
        /// <exception cref="NullReferenceException">Thrown when no image data is found in the input data.</exception>
        /// <exception cref="IncorrectDimensionsException">Thrown when the input does not have the dimensions supplied in ResNet options.</exception>
        private protected NDArray GetInputData(DataViewRowCursor cursor, ValueGetter<MLImage> imageDataGetter, int height, int width)
        {
            var list = new List<byte[,,]>();
            while (cursor.MoveNext())
            {
                MLImage? imageValue = default;
                imageDataGetter(ref imageValue!);

                CheckImage(imageValue, height, width);

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

        private void CheckImage(MLImage image, int height, int width)
        {
            if (image == null)
            {
                throw new NullReferenceException("No image data was found in the input data.");
            }

            if (image.Width != width || image.Height != height)
            {
                throw new IncorrectDimensionsException("Image width and height must be at least 32 pixels.");
            }
        }

        /// <summary>
        /// Converts image into pixel array,
        /// </summary>
        /// <param name="image">Input image.</param>
        /// <returns>2D pixel array.</returns>
        private protected Pixel[,] GetPixelsFromImage(MLImage image)
        {
            var result = new Pixel[image.Height, image.Width];
            int i = 0;
            int j = 0;
            int k = 0;

            while (k < image.Width * image.Height * 4)
            {
                result[i, j] = GetPixelFromImage(image, k);
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
        private protected Pixel GetPixelFromImage(MLImage image, int index)
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
                    throw new UnknownPixelFormatException("Pixel format of input images is unknown. Use either RGBA or BGRA.");
            }
        }

        /// <summary>
        /// Gets labels from input data.
        /// </summary>
        /// <param name="cursor">Cursor pointing to input data.</param>
        /// <param name="labelGetter">Getter for label.</param>
        /// <returns>1D NDArray containing labels.</returns>
        private protected NDArray GetLabels(DataViewRowCursor cursor, ValueGetter<uint> labelGetter, int numberOfClasses)
        {
            uint label = default;

            var labels = new List<uint>();

            while (cursor.MoveNext())
            {
                labelGetter(ref label);
                labels.Add(label);
            }

            var result = new float[labels.Count, numberOfClasses];

            for (int i = 0; i < labels.Count; i++)
            {
                var cls = labels[i];
                for (int j = 0; j < cls; j++)
                {
                    result[i, j] = 0;
                }
                result[i, cls] = 1;
                for (int j = (int)(cls + 1); j < numberOfClasses; j++)
                {
                    result[i, j] = 0;
                }
            }

            return new NDArray(result, new Shape(labels.Count, numberOfClasses));
        }
    }
}
