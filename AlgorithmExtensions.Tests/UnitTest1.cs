using AlgorithmExtensions.Hyperalgorithms;
using AlgorithmExtensions.Scoring;
using Microsoft.ML;
using Microsoft.ML.Data;
using Microsoft.ML.Trainers;
using Microsoft.ML.Trainers.LightGbm;
using Microsoft.ML.Transforms.Text;
using Microsoft.ML.Vision;
using NumSharp.Extensions;
using System.Diagnostics;
using Xunit.Sdk;
using static Microsoft.ML.TrainCatalogBase;

namespace AlgorithmExtensions.Tests
{
    public class UnitTest1
    {
        [Fact]
        public void Test1()
        {
            var context = new MLContext();
            var pipeline = new PipelineTemplate();

            pipeline.Add(new Func<LbfgsPoissonRegressionTrainer.Options, LbfgsPoissonRegressionTrainer>(context.Regression.Trainers.LbfgsPoissonRegression), "lbfgs");
            pipeline.Add(new Func<AveragedPerceptronTrainer.Options, AveragedPerceptronTrainer>(context.BinaryClassification.Trainers.AveragedPerceptron), "perceptron");
            pipeline.Add(new Func<ImageClassificationTrainer.Options, ImageClassificationTrainer>(context.MulticlassClassification.Trainers.ImageClassification), "image");

            var options = new LbfgsPoissonRegressionTrainer.Options();
            
            var paramDict = new Dictionary<string, string[]>();
            paramDict.Add("lbfgs", new string[] { "L1Regularization_1,0", "L1Regularization_2,0" });
            paramDict.Add("perceptron", new string[] { "L2Regularization_0,1", "L2Regularization_0,2" });
            paramDict.Add("image", new string[] { "Epoch_100" });

            var gridSearch = new GridSearchCV<ModelOutput>(context, pipeline, paramDict);
            var result = gridSearch.GeneratePipelinesFromParameters().ToList();
        }

        [Fact]
        public void TestTraining()
        {
            var mlContext = new MLContext();
            IDataView trainingDataView = mlContext.Data.LoadFromTextFile<ModelInput>(
                                            path: @"C:\Users\Oliver\Desktop\creditcard.csv",
                                            hasHeader: true,
                                            separatorChar: ',',
                                            allowQuoting: true,
                                            allowSparse: false);

            var a = mlContext.Data.TrainTestSplit(trainingDataView, 0.5);
            

            var dataProcessPipeline = mlContext.Transforms.Concatenate("Features", new[] { "Time", "V1", "V2", "V3", "V4", "V5", "V6", "V7", "V8", "V9", "V10", "V11", "V12", "V13", "V14", "V15", "V16", "V17", "V18", "V19", "V20", "V21", "V22", "V23", "V24", "V25", "V26", "V27", "V28", "Amount" });

            // Choosing algorithm
            var trainer = mlContext.BinaryClassification.Trainers.LinearSvm();
            // Appending algorithm to pipeline
            var trainingPipeline = dataProcessPipeline.Append(trainer);

            ITransformer model = trainingPipeline.Fit(a.TrainSet);

            var prediction = model.Transform(trainingDataView);
            var metrics = mlContext.BinaryClassification.EvaluateNonCalibrated(prediction);

            Debug.WriteLine("");
        }

        [Fact]
        public void YelpTest()
        {
            var mlContext = new MLContext();
            var dataView = mlContext.Data.LoadFromTextFile<YelpInput>(@"C:\Users\Oliver\Desktop\sentiment labelled sentences\sentiment labelled sentences\yelp_labelled.txt");

            var a = mlContext.Data.TrainTestSplit(dataView, 0.2);

            var dataPipeline = mlContext.Transforms.Text.FeaturizeText("Features", "Text")
                .Append(mlContext.BinaryClassification.Trainers.LinearSvm());

            var transformer = dataPipeline.Fit(a.TrainSet);
            var prediction = transformer.Transform(a.TestSet);

            var metrics = mlContext.BinaryClassification.CrossValidateNonCalibrated(dataView, dataPipeline);

            Debug.WriteLine("");
        }

        [Fact]
        public async Task TestGridSearchOnYelp()
        {
            var mlContext = new MLContext();
            var dataView = mlContext.Data.LoadFromTextFile<YelpInput>(@"C:\Users\Oliver\Desktop\sentiment labelled sentences\sentiment labelled sentences\yelp_labelled.txt");

            //var a = mlContext.Data.CreateEnumerable<YelpInput>(dataView, false).ToArray();

            var pipelineTemplate = new PipelineTemplate();
            var del = new Func<string, string, TextFeaturizingEstimator>(mlContext.Transforms.Text.FeaturizeText);
            var featurizeTextDefaultParameters = new object[] { "Features", "Text" };
            pipelineTemplate.Add(del, "text", featurizeTextDefaultParameters);
            var model = new Func<LinearSvmTrainer.Options, LinearSvmTrainer>(mlContext.BinaryClassification.Trainers.LinearSvm);
            pipelineTemplate.Add(model, "svm");

            var parameters = new Dictionary<string, string[]>();
            parameters.Add("svm", new string[] { "NumberOfIterations_1" });

            var gridSearch = new GridSearchCV<YelOutput>(mlContext, pipelineTemplate, parameters, new AccuracyScoringFunction<YelOutput>(mlContext));
            await gridSearch.Fit(dataView);
        }

        [Fact]
        public async Task TestGridSearch()
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

            var parameters = new Dictionary<string, string[]>();
            parameters.Add("lgbm", new string[] { "NumberOfIterations_1", "NumberOfIterations_100" });

            var gridSearch = new GridSearchCV<ModelOutput>(mlContext, pipelineTamplate, parameters, new AccuracyScoringFunction<ModelOutput>(mlContext));
            await gridSearch.Fit(trainingDataView);
        }

        [Fact]
        public void GenerateEstimatorChain()
        {
            var mlContext = new MLContext();
            var pipelineTamplate = new PipelineTemplate();

            var concatenateDefaultParameters = new object[]
            {
                "Features",
                new string[] { "Time", "V1", "V2", "V3", "V4", "V5", "V6", "V7", "V8", "V9", "V10", "V11", "V12", "V13", "V14", "V15", "V16", "V17", "V18", "V19", "V20", "V21", "V22", "V23", "V24", "V25", "V26", "V27", "V28", "Amount" }
            };
            pipelineTamplate.Add(mlContext.Transforms.Concatenate, "concatenate", concatenateDefaultParameters);
            var parameters = new Dictionary<string, string[]>();
            parameters.Add("lgbm", new string[] { "NumberOfIterations_1", "NumberOfIterations_100" });
            pipelineTamplate.Add(new Func<LightGbmBinaryTrainer.Options, LightGbmBinaryTrainer>(mlContext.BinaryClassification.Trainers.LightGbm), "lgbm");

            var gridSearch = new GridSearchCV<ModelOutput>(mlContext, pipelineTamplate, parameters);
            var result = gridSearch.GeneratePipelinesFromParameters().ToArray();
            Assert.Equal(2, result.Count());
        }
    }
}