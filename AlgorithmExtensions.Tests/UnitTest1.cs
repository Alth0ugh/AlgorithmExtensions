using AlgorithmExtensions.Hyperalgorithms;
using AlgorithmExtensions.Hyperalgorithms.ParameterProviders;
using AlgorithmExtensions.ResNets;
using AlgorithmExtensions.Scoring;
using Google.Protobuf;
using Microsoft.ML;
using Microsoft.ML.Data;
using Microsoft.ML.Trainers;
using Microsoft.ML.Trainers.LightGbm;
using Microsoft.ML.Transforms;
using Microsoft.ML.Transforms.Text;
using Microsoft.ML.Vision;
using System.Diagnostics;
using static Microsoft.ML.Transforms.ValueToKeyMappingEstimator;
using static System.Runtime.InteropServices.JavaScript.JSType;
using AlgorithmExtensions.Extensions;
using AlgorithmExtensions.Exceptions;

namespace AlgorithmExtensions.Tests
{
    public class UnitTest1
    {
        [Fact]
        public async Task Fit_GridSearchWithTwoParameterProviders_ShouldSucceed()
        {
            var mlContext = new MLContext();
            var dataView = mlContext.Data.LoadFromTextFile<YelpInput>(@"C:\Users\Oliver\Desktop\sentiment labelled sentences\sentiment labelled sentences\yelp_labelled.txt");

            var pipelineTemplate = new PipelineTemplate();
            var del = new Func<string, string, TextFeaturizingEstimator>(mlContext.Transforms.Text.FeaturizeText);
            var featurizeTextDefaultParameters = new object[] { "Features", "Text" };
            pipelineTemplate.Add(del, "text", featurizeTextDefaultParameters);
            var model = new Func<LinearSvmTrainer.Options, LinearSvmTrainer>(mlContext.BinaryClassification.Trainers.LinearSvm);
            pipelineTemplate.Add(model, "svm");

            var parameters = new ParameterProviderForModel();
            parameters.Add("svm", new ConstantParameterProvider(nameof(LinearSvmTrainer.Options.NumberOfIterations), 1),
                new StepParameterProvider<float>(nameof(LinearSvmTrainer.Options.Lambda), 1, 100, 10));

            var gridSearch = new GridSearchCV(mlContext, pipelineTemplate, parameters, new FScoringFunctionBinary<YelOutput>(mlContext));
            await gridSearch.Fit(dataView);
        }

        [Fact]
        public async Task Fit_GridSearchWithCorrectParameters_ShouldSucceed()
        {
            var mlContext = new MLContext();

            IDataView trainingDataView = mlContext.Data.LoadFromTextFile<ModelInput>(
                                            path: @"C:\Users\Oliver\Desktop\creditcard.csv",
                                            hasHeader: true,
                                            separatorChar: ',',
                                            allowQuoting: true,
                                            allowSparse: false);

            var pipelineTamplate = new PipelineTemplate();
            var concatenateDefaultParameters = new object[]
            {
                "Features",
                new string[] { "Time", "V1", "V2", "V3", "V4", "V5", "V6", "V7", "V8", "V9", "V10", "V11", "V12", "V13", "V14", "V15", "V16", "V17", "V18", "V19", "V20", "V21", "V22", "V23", "V24", "V25", "V26", "V27", "V28", "Amount" }
            };
            pipelineTamplate.Add(mlContext.Transforms.Concatenate, "concatenate", concatenateDefaultParameters);
            pipelineTamplate.Add(new Func<LightGbmBinaryTrainer.Options, LightGbmBinaryTrainer>(mlContext.BinaryClassification.Trainers.LightGbm), "lgbm");

            var parameters = new ParameterProviderForModel();
            parameters.Add("lgbm", new ConstantParameterProvider(nameof(LightGbmBinaryTrainer.Options.NumberOfIterations), 1, 100));

            var gridSearch = new GridSearchCV(mlContext, pipelineTamplate, parameters, new AccuracyScoringFunction<ModelOutput>(mlContext));
            await gridSearch.Fit(trainingDataView);
        }

        /*
        [Fact]
        public void TrainGithubIssue()
        {
            var mlContext = new MLContext();
            var data = mlContext.Data.LoadFromTextFile<GihubIssue>(@"C:\Users\Oliver\Desktop\issues.tsv", hasHeader: true);

            var pipeline = mlContext.Transforms.Conversion.MapValueToKey(inputColumnName: "Area", outputColumnName: "Label")
                .Append(mlContext.Transforms.Text.FeaturizeText(inputColumnName: "Title", outputColumnName: "TitleFeaturized"))
                .Append(mlContext.Transforms.Text.FeaturizeText(inputColumnName: "Description", outputColumnName: "DescriptionFeaturized"))
                .Append(mlContext.Transforms.Concatenate("Features", "TitleFeaturized", "DescriptionFeaturized"));

            var trainingPipeline = pipeline.Append(mlContext.MulticlassClassification.Trainers.SdcaMaximumEntropy("Label", "Features"));
            var transformer = trainingPipeline.Fit(data);
            var newData = transformer.Transform(data);
            var metrics = mlContext.MulticlassClassification.CrossValidate(data, trainingPipeline);

            Debug.WriteLine("");
        }/*/

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

        [Fact]
        public void GetRow_ResNetMapperWithCorrectColumns_ShouldSucceed()
        {
            var mlContext = new MLContext();
            IEnumerable<ImgData> imgs = LoadImagesFromDirectory(folder: @"C:\Users\Oliver\Desktop\SDNET", useFolderNameAsLabel: true);
            IDataView imgData = mlContext.Data.LoadFromEnumerable(imgs);

            var preprocessingPipeline = mlContext.Transforms.Conversion.MapValueToKey(inputColumnName: "Label", outputColumnName: "LabelKey")
                .Append(mlContext.Transforms.LoadImages(outputColumnName: "Features", imageFolder: @"C:\Users\Oliver\Desktop\SDNET", inputColumnName: "ImagePath"));

            var data = preprocessingPipeline.Fit(imgData).Transform(imgData);

            var options = new Options() { BatchSize = 30, Epochs = 1, Classes = 7, Architecture = ResNetArchitecture.ResNet50, FeatureColumnName = "Features", LabelColumnName = "LabelKey" };

            var resnet = mlContext.MulticlassClassification.Trainers.ResNetClassificator(options);

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
            IEnumerable<ImgData> imgs = LoadImagesFromDirectory(folder: @"C:\Users\Oliver\Desktop\SDNET", useFolderNameAsLabel: true);
            IDataView imgData = mlContext.Data.LoadFromEnumerable(imgs);

            var preprocessingPipeline = mlContext.Transforms.Conversion.MapValueToKey(inputColumnName: "Label", outputColumnName: "LabelKey")
                .Append(mlContext.Transforms.LoadImages(outputColumnName: "Features", imageFolder: @"C:\Users\Oliver\Desktop\SDNET", inputColumnName: "ImagePath"));

            var data = preprocessingPipeline.Fit(imgData).Transform(imgData);

            var options = new Options() { BatchSize = 30, Epochs = 1, Classes = 7, Architecture = ResNetArchitecture.ResNet50, FeatureColumnName = "Features", LabelColumnName = "LabelKey" };

            var resnet = mlContext.MulticlassClassification.Trainers.ResNetClassificator(options);

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
            IEnumerable<ImgData> imgs = LoadImagesFromDirectory(folder: @"C:\Users\Oliver\Desktop\SDNET", useFolderNameAsLabel: true);
            IDataView imgData = mlContext.Data.LoadFromEnumerable(imgs);

            var preprocessingPipeline = mlContext.Transforms.Conversion.MapValueToKey(inputColumnName: "Label", outputColumnName: "LabelKey")
                .Append(mlContext.Transforms.LoadImages(outputColumnName: "Features", imageFolder: @"C:\Users\Oliver\Desktop\SDNET", inputColumnName: "ImagePath"));

            var data = preprocessingPipeline.Fit(imgData).Transform(imgData);

            var options = new Options() { BatchSize = 30, Epochs = 1, Classes = 7, Architecture = ResNetArchitecture.ResNet50, FeatureColumnName = "Features", LabelColumnName = "LabelKey" };

            var resnet = mlContext.MulticlassClassification.Trainers.ResNetClassificator(options);

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
            IEnumerable<ImgData> imgs = LoadImagesFromDirectory(folder: @"C:\Users\Oliver\Desktop\SDNET", useFolderNameAsLabel: true);
            IDataView imgData = mlContext.Data.LoadFromEnumerable(imgs);

            var preprocessingPipeline = mlContext.Transforms.Conversion.MapValueToKey(inputColumnName: "Label", outputColumnName: "LabelKey")
                .Append(mlContext.Transforms.LoadImages(outputColumnName: "Features", imageFolder: @"C:\Users\Oliver\Desktop\SDNET", inputColumnName: "ImagePath"));

            var data = preprocessingPipeline.Fit(imgData).Transform(imgData);

            var options = new Options() { BatchSize = 30, Epochs = 1, Classes = 7, Architecture = ResNetArchitecture.ResNet50, FeatureColumnName = "Features", LabelColumnName = "LabelKey" };

            var resnet = mlContext.MulticlassClassification.Trainers.ResNetClassificator(options);

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
            IEnumerable<ImgData> imgs = LoadImagesFromDirectory(folder: @"C:\Users\Oliver\Desktop\SDNET", useFolderNameAsLabel: true);
            IDataView imgData = mlContext.Data.LoadFromEnumerable(imgs);

            var preprocessingPipeline = mlContext.Transforms.Conversion.MapValueToKey(inputColumnName: "Label", outputColumnName: "LabelKey")
                .Append(mlContext.Transforms.LoadImages(outputColumnName: "Features", imageFolder: @"C:\Users\Oliver\Desktop\SDNET", inputColumnName: "ImagePath"));

            var data = preprocessingPipeline.Fit(imgData).Transform(imgData);

            var options = new Options() { BatchSize = 30, Epochs = 1, Classes = 7, Architecture = ResNetArchitecture.ResNet50, FeatureColumnName = "Features", LabelColumnName = "LabelKey" };

            var resnet = mlContext.MulticlassClassification.Trainers.ResNetClassificator(options);

            var transformer = resnet.Fit(data);

            var mapper = transformer.GetRowToRowMapper(data.Schema);

            Assert.Throws<MissingColumnException>(() => mapper.GetDependencies(imgData.Schema));
        }

        [Fact]
        public void GetRowToRowMapper_TrainedResNet_ShouldSucceed()
        {
            var mlContext = new MLContext();
            IEnumerable<ImgData> imgs = LoadImagesFromDirectory(folder: @"C:\Users\Oliver\Desktop\SDNET", useFolderNameAsLabel: true);
            IDataView imgData = mlContext.Data.LoadFromEnumerable(imgs);

            var preprocessingPipeline = mlContext.Transforms.Conversion.MapValueToKey(inputColumnName: "Label", outputColumnName: "LabelKey")
                .Append(mlContext.Transforms.LoadImages(outputColumnName: "Features", imageFolder: @"C:\Users\Oliver\Desktop\SDNET", inputColumnName: "ImagePath"));

            var data = preprocessingPipeline.Fit(imgData).Transform(imgData);

            var options = new Options() { BatchSize = 30, Epochs = 1, Classes = 7, Architecture = ResNetArchitecture.ResNet50, FeatureColumnName = "Features", LabelColumnName = "LabelKey" };

            var resnet = mlContext.MulticlassClassification.Trainers.ResNetClassificator(options);

            var transformer = resnet.Fit(data);

            _ = transformer.GetRowToRowMapper(data.Schema);
        }

        [Fact]
        public void GetRowToRowMapper_TrainedResNetWrongInputSchema_ShouldThrowException()
        {
            var mlContext = new MLContext();
            IEnumerable<ImgData> imgs = LoadImagesFromDirectory(folder: @"C:\Users\Oliver\Desktop\SDNET", useFolderNameAsLabel: true);
            IDataView imgData = mlContext.Data.LoadFromEnumerable(imgs);

            var preprocessingPipeline = mlContext.Transforms.Conversion.MapValueToKey(inputColumnName: "Label", outputColumnName: "LabelKey")
                .Append(mlContext.Transforms.LoadImages(outputColumnName: "Features", imageFolder: @"C:\Users\Oliver\Desktop\SDNET", inputColumnName: "ImagePath"));

            var data = preprocessingPipeline.Fit(imgData).Transform(imgData);

            var options = new Options() { BatchSize = 30, Epochs = 1, Classes = 7, Architecture = ResNetArchitecture.ResNet50, FeatureColumnName = "Features", LabelColumnName = "LabelKey" };

            var resnet = mlContext.MulticlassClassification.Trainers.ResNetClassificator(options);

            var transformer = resnet.Fit(data);
            Assert.Throws<MissingColumnException>(() => transformer.GetRowToRowMapper(imgData.Schema));
        }

        /*
        [Fact]
        public void TestResNetFunctions()
        {
            var mlContext = new MLContext();
            IEnumerable<ImgData> imgs = LoadImagesFromDirectory(folder: @"C:\Users\Oliver\Desktop\SDNET", useFolderNameAsLabel: true);
            IDataView imgData = mlContext.Data.LoadFromEnumerable(imgs);

            var preprocessingPipeline = mlContext.Transforms.Conversion.MapValueToKey(inputColumnName: "Label", outputColumnName: "LabelKey")
                .Append(mlContext.Transforms.LoadImages(outputColumnName: "Features", imageFolder: @"C:\Users\Oliver\Desktop\SDNET", inputColumnName: "ImagePath"));

            var data = preprocessingPipeline.Fit(imgData).Transform(imgData);

            var resnet = new ResNetTrainer(new Options() { BatchSize = 5, Epochs = 5 });

            var featureColumn = data.Schema["Features"];
            var labelColumn = data.Schema["LabelKey"];

            var cursor1 = data.GetRowCursor(new[] { featureColumn, labelColumn });
            var cursor2 = data.GetRowCursor(new[] { featureColumn, labelColumn });

            var imageDataGetter = cursor1.GetGetter<MLImage>(featureColumn);
            var labelGetter = cursor2.GetGetter<uint>(labelColumn);

            var result = resnet.GetInputData2(cursor1, imageDataGetter);

            var result2 = resnet.GetLabels(cursor2, labelGetter);

            Debug.WriteLine("");
        }
        */
        [Fact]
        public void Fit_Transform_ResNet50_ShouldSucceed()
        {
            var mlContext = new MLContext();
            IEnumerable<ImgData> imgs = LoadImagesFromDirectory(folder: @"C:\Users\Oliver\Desktop\SDNET", useFolderNameAsLabel: true);
            IDataView imgData = mlContext.Data.LoadFromEnumerable(imgs);

            var preprocessingPipeline = mlContext.Transforms.Conversion.MapValueToKey(inputColumnName: "Label", outputColumnName: "LabelKey")
                .Append(mlContext.Transforms.LoadImages(outputColumnName: "Features", imageFolder: @"C:\Users\Oliver\Desktop\SDNET", inputColumnName: "ImagePath"));

            var data = preprocessingPipeline.Fit(imgData).Transform(imgData);

            var options = new Options() { BatchSize = 30, Epochs = 1, Classes = 7, Architecture = ResNetArchitecture.ResNet50, FeatureColumnName = "Features", LabelColumnName = "LabelKey" };

            var resnet = mlContext.MulticlassClassification.Trainers.ResNetClassificator(options);

            resnet.Fit(data).Transform(data);
        }

        [Fact]
        public void GetOutputSchema_TrainedResnet50_ShouldSucceed()
        {
            var mlContext = new MLContext();
            IEnumerable<ImgData> imgs = LoadImagesFromDirectory(folder: @"C:\Users\Oliver\Desktop\SDNET", useFolderNameAsLabel: true);
            IDataView imgData = mlContext.Data.LoadFromEnumerable(imgs);

            var preprocessingPipeline = mlContext.Transforms.Conversion.MapValueToKey(inputColumnName: "Label", outputColumnName: "LabelKey")
                .Append(mlContext.Transforms.LoadImages(outputColumnName: "Features", imageFolder: @"C:\Users\Oliver\Desktop\SDNET", inputColumnName: "ImagePath"));

            var data = preprocessingPipeline.Fit(imgData).Transform(imgData);

            var options = new Options() { BatchSize = 30, Epochs = 1, Classes = 7, Architecture = ResNetArchitecture.ResNet50, FeatureColumnName = "Features", LabelColumnName = "LabelKey" };

            var resnet = mlContext.MulticlassClassification.Trainers.ResNetClassificator(options);

            var transformer = resnet.Fit(data);
            var outputSchema = transformer.GetOutputSchema(data.Schema);

            try
            {
                _ = outputSchema["Prediction"];
                _ = outputSchema["Scores"];
            }
            catch
            {
                Assert.Fail("Columns Prediction or Scores are not present in the output schema.");
            }
        }
    }
}