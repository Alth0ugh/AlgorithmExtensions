using AlgorithmExtensions.Hyperalgorithms;
using AlgorithmExtensions.Hyperalgorithms.ParameterProviders;
using AlgorithmExtensions.Scoring;
using Microsoft.ML;
using Microsoft.ML.Trainers;
using Microsoft.ML.Trainers.LightGbm;
using Microsoft.ML.Transforms.Text;
using AlgorithmExtensions.Exceptions;
using Microsoft.ML.Transforms;
using static Microsoft.ML.Transforms.ValueToKeyMappingEstimator;
using Microsoft.ML.Trainers.FastTree;
using AlgorithmExtensions.Extensions;
using AlgorithmExtensions.ResNets;

namespace AlgorithmExtensions.Tests
{
    public class GridSearchTests
    {
        private IDataView GetInputData(MLContext mlContext)
        {
            return mlContext.Data.LoadFromTextFile<CreditCardInput>(
                                            path: @"..\..\..\TestData\creditcard.csv",
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
            var dataView = mlContext.Data.LoadFromTextFile<YelpInput>(@"..\..\..\TestData\yelp_labelled.txt");

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

        [Fact]
        public async Task Fit_GridSearchMulticlassClassification_ShouldSucceed()
        {
            var mlContext = new MLContext();
            var data = mlContext.Data.LoadFromTextFile<GihubIssue>(@"C:\Users\Oliver\Desktop\issues.tsv", hasHeader: true);

            var pipelineTemplate = new PipelineTemplate();

            var mapValToKey = new Func<string, string, int, KeyOrdinality, bool, IDataView, ValueToKeyMappingEstimator>(mlContext.Transforms.Conversion.MapValueToKey);

            pipelineTemplate.Add(mapValToKey, "key", new object[] { "Label", "Area", 1000000, KeyOrdinality.ByOccurrence, false, null });
            var featurizeText = new Func<string, string, TextFeaturizingEstimator>(mlContext.Transforms.Text.FeaturizeText);
            pipelineTemplate.Add(featurizeText, "featurize1", new object[] { "TitleFeaturized", "Title" });
            pipelineTemplate.Add(featurizeText, "featurize2", new object[] { "DescriptionFeaturized", "Description" });
            pipelineTemplate.Add(mlContext.Transforms.Concatenate, "concatenate", new object[] { "Features", new string[] { "TitleFeaturized", "DescriptionFeaturized" } });

            var model = new Func<SdcaMaximumEntropyMulticlassTrainer.Options, SdcaMaximumEntropyMulticlassTrainer>(mlContext.MulticlassClassification.Trainers.SdcaMaximumEntropy);
            pipelineTemplate.Add(model, "model");

            var parameters = new ParameterProviderForModel();
            parameters.Add("model", new ConstantParameterProvider(nameof(SdcaMaximumEntropyMulticlassTrainer.Options.MaximumNumberOfIterations), 1, 3),
                new ConstantParameterProvider(nameof(SdcaMaximumEntropyMulticlassTrainer.Options.L1Regularization), null, 0.2f));

            var f_score = new FScoringFunctionMulticlass<GitHubIssueOutput>(mlContext, 22, true);

            var gridSearch = new GridSearchCV(mlContext, pipelineTemplate, parameters, new AccuracyScoringFunction<GitHubIssueOutput>(mlContext));
            await gridSearch.Fit(data);

            Assert.Equal(3, (int)gridSearch.BestParameters!["model"]
                .Where(x => x.Name == nameof(SdcaMaximumEntropyMulticlassTrainer.Options.MaximumNumberOfIterations))
                .Single().Value);
            Assert.Equal(0.2f, (float)gridSearch.BestParameters!["model"]
                .Where(x => x.Name == nameof(SdcaMaximumEntropyMulticlassTrainer.Options.L1Regularization))
                .Single().Value);
        }

        [Fact]
        public async Task Fit_GridSearchWithRegression_ShouldSucceed()
        {
            var mlContext = new MLContext();
            IDataView dataView = mlContext.Data.LoadFromTextFile<TaxiTrip>(@"..\..\..\TestData\taxi-fare-train.csv", hasHeader: true, separatorChar: ',');

            var pipelineTemplate = new PipelineTemplate();
            pipelineTemplate.Add(mlContext.Transforms.CopyColumns, "copy", "Label", "FareAmount");
            var oneHotDelegate = new Func<string, string, OneHotEncodingEstimator.OutputKind, int, KeyOrdinality, IDataView, OneHotEncodingEstimator>(mlContext.Transforms.Categorical.OneHotEncoding);
            pipelineTemplate.Add(oneHotDelegate, "oneHot1", "VendorIdEncoded", "VendorId");
            pipelineTemplate.Add(oneHotDelegate, "oneHot2", "RateCodeEncoded", "RateCode");
            pipelineTemplate.Add(oneHotDelegate, "oneHot3", "PaymentTypeEncoded", "PaymentType");
            pipelineTemplate.Add(mlContext.Transforms.Concatenate, "concat", new object[] { "Features", new string[] { "VendorIdEncoded", "RateCodeEncoded", "PassengerCount", "TripDistance", "PaymentTypeEncoded" } });
            var fastTreeDelegate = new Func<FastTreeRegressionTrainer.Options, FastTreeRegressionTrainer>(mlContext.Regression.Trainers.FastTree);
            pipelineTemplate.Add(fastTreeDelegate, "fastTree");

            var parameters = new ParameterProviderForModel();
            parameters.Add("fasTree", new ConstantParameterProvider(nameof(FastTreeRegressionTrainer.Options.LearningRate), 0.2d));

            var gridSeach = new GridSearchCV(mlContext, pipelineTemplate, parameters, new MeanSquareErrorScoringFunction<TaxiPrediction>(mlContext));

            await gridSeach.Fit(dataView);
        }

        [Fact]
        public async Task Fit_GridSearchWithResnet_ShouldSucceed()
        {
            var mlContext = new MLContext();
            IDataView dataView = ResNetTests.GetInputData(mlContext, ResNetTests.CorrectSDNET);

            var pipelineTemplate = new PipelineTemplate();
            pipelineTemplate.Add(mlContext.MulticlassClassification.Trainers.ResNetClassificator, "resnet", new Options() { LabelColumnName = "LabelKey", Height = 256, Width = 256 });

            var parameters = new ParameterProviderForModel();
            parameters.Add("resnet", new ConstantParameterProvider(nameof(Options.Epochs), 1, 3));

            var gridSearch = new GridSearchCV(mlContext, pipelineTemplate, parameters, new AccuracyScoringFunction<ResNetPrediction>(mlContext), multipleThreads: false);
            await gridSearch.Fit(dataView);
        }
    }
}