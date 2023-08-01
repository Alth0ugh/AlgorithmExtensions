using AlgorithmExtensions.ResNets;
using AlgorithmExtensions.Tests;
using Microsoft.ML;
using System.Data;
using System.Reflection.Metadata.Ecma335;
using Tensorflow;
using Tensorflow.NumPy;
using Tensorflow.Operations.Initializers;
using static Tensorflow.Binding;
using static Tensorflow.KerasApi;

namespace ConsoleApp1
{
    internal class Program
    {
        static void Main(string[] args)
        {
            var mlContext = new MLContext();
            IEnumerable<ImgData> imgs = LoadImagesFromDirectory(folder: @"C:\Users\Oliver\Desktop\SDNET", useFolderNameAsLabel: true);
            IDataView imgData = mlContext.Data.LoadFromEnumerable(imgs);
            imgData = mlContext.Data.ShuffleRows(imgData);

            var preprocessingPipeline = mlContext.Transforms.Conversion.MapValueToKey(inputColumnName: "Label", outputColumnName: "LabelKey")
                .Append(mlContext.Transforms.LoadImages(outputColumnName: "Features", imageFolder: @"C:\Users\Oliver\Desktop\SDNET", inputColumnName: "ImagePath"));
            //var preprocessingPipeline = mlContext.Transforms.LoadImages(outputColumnName: "Features", imageFolder: @"C:\Users\Oliver\Desktop\SDNET", inputColumnName: "ImagePath");

            var data = preprocessingPipeline.Fit(imgData).Transform(imgData);
            var split = mlContext.Data.TrainTestSplit(data, 0.5, seed: 42);

            var resnet = new ResNetTrainer(new Options() { BatchSize = -1, Epochs = 100, Classes = 10, Architecture = ResNetArchitecture.ResNet50, FeatureColumnName = "Features", LabelColumnName = "LabelKey" }, mlContext);

            var ((x_train, y_train), (x_test, y_test)) = keras.datasets.cifar10.load_data();
            x_train = x_train / 255.0f;
            x_test = x_test / 255.0f;
            y_train = GetLabels(y_train);

            resnet.Fit(x_train[new Slice(0, 1000)], y_train[new Slice(0, 1000)]).Transform(x_test);
            //resnet.Fit(split.TrainSet).Transform(split.TestSet);
        }

        static NDArray GetLabels(NDArray y)
        {
            var result = new float[y.shape[0], 10];

            for (int i = 0; i < y.shape[0]; i++)
            {
                var cls = (int)y[i];
                for (int j = 0; j < cls; j++)
                {
                    result[i, j] = 0;
                }
                result[i, cls] = 1;
                for (int j = (int)(cls + 1); j < 10; j++)
                {
                    result[i, j] = 0;
                }
            }

            return new NDArray(result, new Shape(y.shape[0], 10));
        }

        public static IEnumerable<ImgData> LoadImagesFromDirectory(string folder, bool useFolderNameAsLabel = true)
        {
            //get all file paths from the subdirectories
            var files = Directory.GetFiles(folder, "*", searchOption:
            SearchOption.AllDirectories);
            //iterate through each file
            foreach (var file in files)
            {
                //Image Classification API supports .jpg and .png formats; check img formats
                if ((Path.GetExtension(file) != ".jpg") &&
                 (Path.GetExtension(file) != ".png"))
                    continue;
                //store filename in a variable, say ‘label’
                var label = Path.GetFileName(file);
                /* If the useFolderNameAsLabel parameter is set to true, then name 
                   of parent directory of the image file is used as the label. Else label is expected to be the file name or a a prefix of the file name. */
                if (useFolderNameAsLabel)
                    label = Directory.GetParent(file).Name;
                else
                {
                    for (int index = 0; index < label.Length; index++)
                    {
                        if (!char.IsLetter(label[index]))
                        {
                            label = label.Substring(0, index);
                            break;
                        }
                    }
                }
                //create a new instance of ImgData()
                yield return new ImgData()
                {
                    ImagePath = file,
                    Label = label
                };
            }
        }

    }
}