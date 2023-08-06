using AlgorithmExtensions.Hyperalgorithms;
using AlgorithmExtensions.Hyperalgorithms.ParameterProviders;
using AlgorithmExtensions.Scoring;
using Microsoft.ML;
using Microsoft.ML.Trainers;
using Microsoft.ML.Trainers.LightGbm;
using Microsoft.ML.Transforms.Text;
using AlgorithmExtensions.Exceptions;
using System.Diagnostics;

namespace AlgorithmExtensions.Tests
{
    public class GridSearchTests
    {
        private IDataView GetInputData(MLContext mlContext)
        {
            return mlContext.Data.LoadFromTextFile<ModelInput>(
                                            path: @"C:\Users\Oliver\Desktop\creditcard.csv",
                                            hasHeader: true,
                                            separatorChar: ',',
                                            allowQuoting: true,
                                            allowSparse: false);
        }

        private string[] GetColumnConcatenation()
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
            parameters.Add("svm", new ConstantParameterProvider(nameof(LinearSvmTrainer.Options.NumberOfIterations), 1, 2),
                new GeometricParameterProvider<float>(nameof(LinearSvmTrainer.Options.Lambda), 0.1f, 3, 0.1f));

            var gridSearch = new GridSearchCV(mlContext, pipelineTemplate, parameters, new FScoringFunctionBinary<YelOutput>(mlContext));
            await gridSearch.Fit(dataView);

            Assert.Equal(2, (int)gridSearch.BestParameters!["svm"]
                .Where(x => x.Name == nameof(LinearSvmTrainer.Options.NumberOfIterations))
                .Single().Value);

            Assert.Equal(0.1f, (float)gridSearch.BestParameters["svm"]
                .Where(x => x.Name == nameof(LinearSvmTrainer.Options.Lambda))
                .Single().Value);

            Assert.True(gridSearch.BestEstimator is not null);
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

            Assert.Equal(100, (int)gridSearch.BestParameters!["lgbm"]
                .Where(x => x.Name == nameof(LightGbmBinaryTrainer.Options.NumberOfIterations))
                .Single().Value);
            Assert.True(gridSearch.BestEstimator is not null);
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

        [Fact]
        public async Task Fit_GridSearchWithIncorrectCreationalDelegate_ShouldThrowException()
        {
            var mlContext = new MLContext();

            IDataView trainingDataView = GetInputData(mlContext);

            var pipelineTamplate = new PipelineTemplate();
            var concatenateDefaultParameters = new object[]
            {
                "Features",
                GetColumnConcatenation()
            };
            pipelineTamplate.Add(Function, "function");
            pipelineTamplate.Add(new Func<LightGbmBinaryTrainer.Options, LightGbmBinaryTrainer>(mlContext.BinaryClassification.Trainers.LightGbm), "lgbm");

            var parameters = new ParameterProviderForModel();
            parameters.Add("lgbm", new ConstantParameterProvider(nameof(LightGbmBinaryTrainer.Options.NumberOfIterations), 1));

            var gridSearch = new GridSearchCV(mlContext, pipelineTamplate, parameters, new AccuracyScoringFunction<ModelOutput>(mlContext));
            await Assert.ThrowsAsync<IncorrectCreationalDelegateException>(async () => await gridSearch.Fit(trainingDataView));
        }

        private void Function()
        {

        }

        [Fact]
        public async Task Fit_GridSearchWithIncorrectArgumentCount_ShouldThowException()
        {
            var mlContext = new MLContext();

            IDataView trainingDataView = GetInputData(mlContext);

            var pipelineTamplate = new PipelineTemplate();
            var concatenateDefaultParameters = new object[]
            {
                "Features",
                GetColumnConcatenation(),
                "shouldNotBeHere"
            };
            pipelineTamplate.Add(mlContext.Transforms.Concatenate, "concatenate", concatenateDefaultParameters);
            pipelineTamplate.Add(new Func<LightGbmBinaryTrainer.Options, LightGbmBinaryTrainer>(mlContext.BinaryClassification.Trainers.LightGbm), "lgbm");

            var parameters = new ParameterProviderForModel();
            parameters.Add("lgbm", new ConstantParameterProvider(nameof(LightGbmBinaryTrainer.Options.NumberOfIterations), 1, 100));

            var gridSearch = new GridSearchCV(mlContext, pipelineTamplate, parameters, new AccuracyScoringFunction<ModelOutput>(mlContext));
            await Assert.ThrowsAsync<IncorrectCreationalDelegateException>(async () => await gridSearch.Fit(trainingDataView));
        }

        [Fact]
        public async Task Fit_GridSearchWithoutAllDefaultParametersInTransformer_ShouldThrowException()
        {
            var mlContext = new MLContext();

            var data = GetInputData(mlContext);

            var pipelineTamplate = new PipelineTemplate();
            pipelineTamplate.Add(mlContext.Transforms.Concatenate, "concatenate");
            pipelineTamplate.Add(new Func<LightGbmBinaryTrainer.Options, LightGbmBinaryTrainer>(mlContext.BinaryClassification.Trainers.LightGbm), "lgbm");

            var parameters = new ParameterProviderForModel();
            parameters.Add("lgbm", new ConstantParameterProvider(nameof(LightGbmBinaryTrainer.Options.NumberOfIterations), 1, 100));

            var gridSearch = new GridSearchCV(mlContext, pipelineTamplate, parameters, new AccuracyScoringFunction<ModelOutput>(mlContext));

            await Assert.ThrowsAsync<IncorrectCreationalDelegateException>(async () => await gridSearch.Fit(data));
        }

        [Fact]
        public void GetParameterValues_GeometricParameterProvider_ShouldReturnArrayOfNumbers()
        {
            var provider = new GeometricParameterProvider<float>("name", 0.1f, 3, 0.1f);
            var result = provider.GetParameterValues();

            var epsilon = 0.00001f;

            Assert.True(Math.Abs((float)result[0] - 0.1f) < epsilon);
            Assert.True(Math.Abs((float)result[1] - 0.01f) < epsilon);
            Assert.True(Math.Abs((float)result[2] - 0.001f) < epsilon);
            Assert.Equal(3, result.Length);
        }

        [Fact]
        public void GetParameterValues_StepParameterProvider_ShouldReturnArrayOfNumbers()
        {
            var provider = new StepParameterProvider<int>("name", 1, 5, 2);
            var result = provider.GetParameterValues();

            Assert.Equal(1, (int)result[0]);
            Assert.Equal(3, (int)result[1]);
            Assert.Equal(5, (int)result[2]);
            Assert.Equal(3, result.Length);
        }

        [Fact]
        public void GetParameterValues_ConstantParameterProvider_ShouldReturnArrayOfNumbers()
        {
            var provider = new ConstantParameterProvider("name", 1, 2, 3);
            var result = provider.GetParameterValues();

            Assert.Equal(1, (int)result[0]);
            Assert.Equal(2, (int)result[1]);
            Assert.Equal(3, (int)result[2]);
            Assert.Equal(3, result.Length);
        }

        [Fact]
        public async Task Fit_GridSearchWithMutlipleProvidersForOneParameter_ShouldThrowException()
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
            parameters.Add("lgbm", new ConstantParameterProvider(nameof(LightGbmBinaryTrainer.Options.NumberOfIterations), 1, 100),
                new GeometricParameterProvider<int>(nameof(LightGbmBinaryTrainer.Options.NumberOfIterations), 1, 3, 10));

            var gridSearch = new GridSearchCV(mlContext, pipelineTamplate, parameters, new AccuracyScoringFunction<ModelOutput>(mlContext));
            await Assert.ThrowsAsync<UniqueValueException>(async () => await gridSearch.Fit(trainingDataView));
        }
    }
}