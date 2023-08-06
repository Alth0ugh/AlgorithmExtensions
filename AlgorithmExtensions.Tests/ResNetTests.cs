using AlgorithmExtensions.Exceptions;
using AlgorithmExtensions.Extensions;
using AlgorithmExtensions.ResNets;
using Microsoft.ML;
using System.Diagnostics;
using System.Text;

namespace AlgorithmExtensions.Tests
{
    public class ResNetTests
    {
        private const string _correctSDNET = @"C:\Users\Oliver\Desktop\SDNET";
        private const string _incorrectSDNET = @"C:\Users\Oliver\Desktop\IncorrectSDNET";

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
                    label = Directory.GetParent(file)!.Name;
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

        public ResNetTrainer GetResnet(MLContext mlContext)
        {
            var options = new Options() { BatchSize = 30, Epochs = 1, Classes = 7, Height = 256, Width = 256, Architecture = ResNetArchitecture.ResNet50, FeatureColumnName = "Features", LabelColumnName = "LabelKey" };

            var resnet = mlContext.MulticlassClassification.Trainers.ResNetClassificator(options);
            return resnet;
        }

        public IDataView GetInputData(MLContext mlContext, string path)
        {
            IEnumerable<ImgData> imgs = LoadImagesFromDirectory(folder: path, useFolderNameAsLabel: true);
            IDataView imgData = mlContext.Data.LoadFromEnumerable(imgs);

            var preprocessingPipeline = mlContext.Transforms.Conversion.MapValueToKey(inputColumnName: "Label", outputColumnName: "LabelKey")
                .Append(mlContext.Transforms.LoadImages(outputColumnName: "Features", imageFolder: path, inputColumnName: "ImagePath"));

            var data = preprocessingPipeline.Fit(imgData).Transform(imgData);
            return data;
        }

        [Fact]
        public void GetRow_ResNetMapperWithCorrectColumns_ShouldSucceed()
        {
            var mlContext = new MLContext();

            var data = GetInputData(mlContext, _correctSDNET);
            var resnet = GetResnet(mlContext);

            var transformer = resnet.Fit(data);

            var mapper = transformer.GetRowToRowMapper(data.Schema);
            var cursor = data.GetRowCursor(data.Schema);
            cursor.MoveNext();
            _ = mapper.GetRow(cursor, data.Schema);
        }

        [Fact]
        public void GetRow_ResNetMapperWithIncorrectColumns_ShouldThrowException()
        {
            var mlContext = new MLContext();
            IEnumerable<ImgData> imgs = LoadImagesFromDirectory(folder: _correctSDNET, useFolderNameAsLabel: true);
            IDataView imgData = mlContext.Data.LoadFromEnumerable(imgs);

            var preprocessingPipeline = mlContext.Transforms.Conversion.MapValueToKey(inputColumnName: "Label", outputColumnName: "LabelKey")
                .Append(mlContext.Transforms.LoadImages(outputColumnName: "Features", imageFolder: _correctSDNET, inputColumnName: "ImagePath"));

            var data = preprocessingPipeline.Fit(imgData).Transform(imgData);

            var resnet = GetResnet(mlContext);

            var transformer = resnet.Fit(data);

            var mapper = transformer.GetRowToRowMapper(data.Schema);
            var cursor = imgData.GetRowCursor(imgData.Schema);
            cursor.MoveNext();
            Assert.Throws<MissingColumnException>(() => mapper.GetRow(cursor, data.Schema));
        }

        [Fact]
        public void GetRow_ResNetMapperWithCorrectColumnsWithInactiveFeatureColumn_ShouldThrowException()
        {
            var mlContext = new MLContext();
            IEnumerable<ImgData> imgs = LoadImagesFromDirectory(folder: _correctSDNET, useFolderNameAsLabel: true);
            IDataView imgData = mlContext.Data.LoadFromEnumerable(imgs);

            var preprocessingPipeline = mlContext.Transforms.Conversion.MapValueToKey(inputColumnName: "Label", outputColumnName: "LabelKey")
                .Append(mlContext.Transforms.LoadImages(outputColumnName: "Features", imageFolder: _correctSDNET, inputColumnName: "ImagePath"));

            var data = preprocessingPipeline.Fit(imgData).Transform(imgData);

            var resnet = GetResnet(mlContext);

            var transformer = resnet.Fit(data);

            var mapper = transformer.GetRowToRowMapper(data.Schema);
            var cursor = data.GetRowCursor(data.Schema);
            cursor.MoveNext();
            Assert.Throws<MissingColumnException>(() => mapper.GetRow(cursor, imgData.Schema));
        }

        [Fact]
        public void GetDependencies_ResNetMapperWithCorrectColumns_ShouldSucceed()
        {
            var mlContext = new MLContext();

            var data = GetInputData(mlContext, _correctSDNET);
            var resnet = GetResnet(mlContext);

            var transformer = resnet.Fit(data);

            var mapper = transformer.GetRowToRowMapper(data.Schema);

            var resultingColumns = mapper.GetDependencies(data.Schema);
            var predictionColumn = from column in resultingColumns
                                   where column.Name == "Features"
                                   select column;
            Assert.Single(predictionColumn);
        }

        [Fact]
        public void GetDependencies_ResNetMapperWithIncorrectColumns_ShouldThrowException()
        {
            var mlContext = new MLContext();
            IEnumerable<ImgData> imgs = LoadImagesFromDirectory(folder: _correctSDNET, useFolderNameAsLabel: true);
            IDataView imgData = mlContext.Data.LoadFromEnumerable(imgs);

            var preprocessingPipeline = mlContext.Transforms.Conversion.MapValueToKey(inputColumnName: "Label", outputColumnName: "LabelKey")
                .Append(mlContext.Transforms.LoadImages(outputColumnName: "Features", imageFolder: _correctSDNET, inputColumnName: "ImagePath"));

            var data = preprocessingPipeline.Fit(imgData).Transform(imgData);

            var resnet = GetResnet(mlContext);

            var transformer = resnet.Fit(data);

            var mapper = transformer.GetRowToRowMapper(data.Schema);

            Assert.Throws<MissingColumnException>(() => mapper.GetDependencies(imgData.Schema));
        }

        [Fact]
        public void GetRowToRowMapper_TrainedResNet_ShouldSucceed()
        {
            var mlContext = new MLContext();

            var data = GetInputData(mlContext, _correctSDNET);
            var resnet = GetResnet(mlContext);

            var transformer = resnet.Fit(data);

            _ = transformer.GetRowToRowMapper(data.Schema);
        }

        [Fact]
        public void GetRowToRowMapper_TrainedResNetWrongInputSchema_ShouldThrowException()
        {
            var mlContext = new MLContext();
            IEnumerable<ImgData> imgs = LoadImagesFromDirectory(folder: _correctSDNET, useFolderNameAsLabel: true);
            IDataView imgData = mlContext.Data.LoadFromEnumerable(imgs);

            var preprocessingPipeline = mlContext.Transforms.Conversion.MapValueToKey(inputColumnName: "Label", outputColumnName: "LabelKey")
                .Append(mlContext.Transforms.LoadImages(outputColumnName: "Features", imageFolder: _correctSDNET, inputColumnName: "ImagePath"));

            var data = preprocessingPipeline.Fit(imgData).Transform(imgData);

            var resnet = GetResnet(mlContext);

            var transformer = resnet.Fit(data);
            Assert.Throws<MissingColumnException>(() => transformer.GetRowToRowMapper(imgData.Schema));
        }

        [Fact]
        public void Fit_Transform_ResNet50_ShouldSucceed()
        {
            var mlContext = new MLContext();

            var data = GetInputData(mlContext, _correctSDNET);
            var resnet = GetResnet(mlContext);

            resnet.Fit(data).Transform(data);
        }

        [Fact]
        public void GetOutputSchema_TrainedResnet50_ShouldSucceed()
        {
            var mlContext = new MLContext();

            var data = GetInputData(mlContext, _correctSDNET);
            var resnet = GetResnet(mlContext);

            var transformer = resnet.Fit(data);
            var outputSchema = transformer.GetOutputSchema(data.Schema);

            try
            {
                _ = outputSchema["Prediction"];
            }
            catch
            {
                Assert.Fail("Column Prediction is not present in the output schema.");
            }
            Assert.True(data.Schema.Count == outputSchema.Count - 1);
        }

        [Fact]
        public void GetOutputSchema_ResnetTrainer_ShouldSucceed()
        {
            var mlContext = new MLContext();

            var data = GetInputData(mlContext, _correctSDNET);
            var resnet = GetResnet(mlContext);

            var transformer = resnet.Fit(data);
            var outputSchema = transformer.GetOutputSchema(data.Schema);

            try
            {
                _ = outputSchema["Prediction"];
            }
            catch
            {
                Assert.Fail("Column Prediction is not present in the output schema.");
            }
            Assert.True(data.Schema.Count == outputSchema.Count - 1);
        }

        [Fact]
        public void Fit_Transform_ResNet50WithIncorrectColumnName_ShouldThrowException()
        {
            var mlContext = new MLContext();

            var data = GetInputData(mlContext, _correctSDNET);
            var options = new Options() { BatchSize = 30, Epochs = 1, Classes = 7, Height = 256, Width = 256, Architecture = ResNetArchitecture.ResNet50, FeatureColumnName = "MissingFeatures", LabelColumnName = "LabelKey" };

            var resnet = mlContext.MulticlassClassification.Trainers.ResNetClassificator(options);

            Assert.Throws<MissingColumnException>(() => resnet.Fit(data));
        }

        [Fact]
        public void Fit_Transform_ResNet50WithIncorrectFeatureDataType_ShouldThrowException()
        {
            var mlContext = new MLContext();

            var data = GetInputData(mlContext, _correctSDNET);
            var options = new Options() { BatchSize = 30, Epochs = 1, Classes = 7, Height = 256, Width = 256, Architecture = ResNetArchitecture.ResNet50, FeatureColumnName = "ImagePath", LabelColumnName = "LabelKey" };

            var resnet = mlContext.MulticlassClassification.Trainers.ResNetClassificator(options);

            Assert.Throws<TypeMismatchException>(() => resnet.Fit(data));
        }

        [Fact]
        public void Fit_ResNetWithIncorrectInputDimensions_ShouldThrowException()
        {
            var mlContext = new MLContext();

            var data = GetInputData(mlContext, _correctSDNET);
            var options = new Options() { BatchSize = 30, Epochs = 1, Classes = 7, Width = 10, Height = 10, Architecture = ResNetArchitecture.ResNet50, FeatureColumnName = "Features", LabelColumnName = "LabelKey" };

            var resnet = mlContext.MulticlassClassification.Trainers.ResNetClassificator(options);
            Assert.Throws<IncorrectDimensionsException>(() => resnet.Fit(data));
        }

        [Fact]
        public void Fit_ResNetWithIncorrectPictureDimensionsInDataset_ShouldThrowException()
        {
            var mlContext = new MLContext();

            var data = GetInputData(mlContext, _incorrectSDNET);
            var resnet = GetResnet(mlContext);

            Assert.Throws<IncorrectDimensionsException>(() => resnet.Fit(data));
        }
    }
}
