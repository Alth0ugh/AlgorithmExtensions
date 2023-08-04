using AlgorithmExtensions.Hyperalgorithms;
using AlgorithmExtensions.Hyperalgorithms.ParameterProviders;
using AlgorithmExtensions.ResNets;
using AlgorithmExtensions.Scoring;
using Microsoft.ML;
using Microsoft.ML.Trainers;
using Microsoft.ML.Trainers.LightGbm;
using Microsoft.ML.Transforms.Text;
using AlgorithmExtensions.Extensions;
using AlgorithmExtensions.Exceptions;

namespace AlgorithmExtensions.Tests
{
    public class GridSearchTests
    {
        public IDataView GetInputData(MLContext mlContext)
        {
            return mlContext.Data.LoadFromTextFile<ModelInput>(
                                            path: @"C:\Users\Oliver\Desktop\creditcard.csv",
                                            hasHeader: true,
                                            separatorChar: ',',
                                            allowQuoting: true,
                                            allowSparse: false);
        }

        public string[] GetColumnConcatenation()
        {
            return new string[] { "Time", "V1", "V2", "V3", "V4", "V5", "V6", "V7", "V8", "V9", "V10", "V11", "V12", "V13", "V14", "V15", "V16", "V17", "V18", "V19", "V20", "V21", "V22", "V23", "V24", "V25", "V26", "V27", "V28", "Amount" };
        }

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

            IDataView trainingDataView = GetInputData(mlContext);

            var pipelineTamplate = new PipelineTemplate();
            var concatenateDefaultParameters = new object[]
            {
                "Features",
                GetColumnConcatenation()
            };
            pipelineTamplate.Add(mlContext.Transforms.Concatenate, "concatenate", concatenateDefaultParameters);
            pipelineTamplate.Add(new Func<LightGbmBinaryTrainer.Options, LightGbmBinaryTrainer>(mlContext.BinaryClassification.Trainers.LightGbm), "lgbm");

            var parameters = new ParameterProviderForModel();
            parameters.Add("lgbm", new ConstantParameterProvider(nameof(LightGbmBinaryTrainer.Options.NumberOfIterations), 1, 100));

            var gridSearch = new GridSearchCV(mlContext, pipelineTamplate, parameters, new AccuracyScoringFunction<ModelOutput>(mlContext));
            await gridSearch.Fit(trainingDataView);
        }

        [Fact]
        public async Task Fit_GridSearchWithIncorrectParameterName_ShouldThrowException()
        {
            var mlContext = new MLContext();

            IDataView trainingDataView = GetInputData(mlContext);

            var pipelineTamplate = new PipelineTemplate();
            var concatenateDefaultParameters = new object[]
            {
                "Features",
                GetColumnConcatenation()
            };
            pipelineTamplate.Add(mlContext.Transforms.Concatenate, "concatenate", concatenateDefaultParameters);
            pipelineTamplate.Add(new Func<LightGbmBinaryTrainer.Options, LightGbmBinaryTrainer>(mlContext.BinaryClassification.Trainers.LightGbm), "lgbm");

            var parameters = new ParameterProviderForModel();
            parameters.Add("lgbm", new ConstantParameterProvider("abcd", 1, 100));

            var gridSearch = new GridSearchCV(mlContext, pipelineTamplate, parameters, new AccuracyScoringFunction<ModelOutput>(mlContext));
            await Assert.ThrowsAsync<IncorrectOptionParameterException>(async () => await gridSearch.Fit(trainingDataView));
        }

        [Fact]
        public async Task Fit_GridSearchWithIncorrectModelName_ShouldThrowException()
        {
            var mlContext = new MLContext();

            IDataView trainingDataView = GetInputData(mlContext);

            var pipelineTamplate = new PipelineTemplate();
            var concatenateDefaultParameters = new object[]
            {
                "Features",
                GetColumnConcatenation()
            };
            pipelineTamplate.Add(mlContext.Transforms.Concatenate, "concatenate", concatenateDefaultParameters);
            pipelineTamplate.Add(new Func<LightGbmBinaryTrainer.Options, LightGbmBinaryTrainer>(mlContext.BinaryClassification.Trainers.LightGbm), "lgbm");

            var parameters = new ParameterProviderForModel();
            parameters.Add("abcd", new ConstantParameterProvider(nameof(LightGbmBinaryTrainer.Options.NumberOfIterations), 1, 100));

            var gridSearch = new GridSearchCV(mlContext, pipelineTamplate, parameters, new AccuracyScoringFunction<ModelOutput>(mlContext));
            await gridSearch.Fit(trainingDataView);
        }

        [Fact]
        public async Task Fit_GridSearchWithNoParameters_ShouldThrowException()
        {
            var mlContext = new MLContext();

            IDataView trainingDataView = GetInputData(mlContext);

            var pipelineTamplate = new PipelineTemplate();
            var concatenateDefaultParameters = new object[]
            {
                "Features",
                GetColumnConcatenation()
            };
            pipelineTamplate.Add(mlContext.Transforms.Concatenate, "concatenate", concatenateDefaultParameters);
            pipelineTamplate.Add(new Func<LightGbmBinaryTrainer.Options, LightGbmBinaryTrainer>(mlContext.BinaryClassification.Trainers.LightGbm), "lgbm");

            var parameters = new ParameterProviderForModel();

            var gridSearch = new GridSearchCV(mlContext, pipelineTamplate, parameters, new AccuracyScoringFunction<ModelOutput>(mlContext));
            await Assert.ThrowsAsync<ParametersMissingException>(async () => await gridSearch.Fit(trainingDataView));
        }

        [Fact]
        public async Task Fit_GridSearchWithEmptyParameters_ShouldThrowException()
        {
            var mlContext = new MLContext();

            IDataView trainingDataView = GetInputData(mlContext);

            var pipelineTamplate = new PipelineTemplate();
            var concatenateDefaultParameters = new object[]
            {
                "Features",
                GetColumnConcatenation()
            };
            pipelineTamplate.Add(mlContext.Transforms.Concatenate, "concatenate", concatenateDefaultParameters);
            pipelineTamplate.Add(new Func<LightGbmBinaryTrainer.Options, LightGbmBinaryTrainer>(mlContext.BinaryClassification.Trainers.LightGbm), "lgbm");

            var parameters = new ParameterProviderForModel();
            parameters.Add("lgbm", new IParameterProvider[0]);

            var gridSearch = new GridSearchCV(mlContext, pipelineTamplate, parameters, new AccuracyScoringFunction<ModelOutput>(mlContext));
            await Assert.ThrowsAsync<ParametersMissingException>(async () => await gridSearch.Fit(trainingDataView));
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
    }
}