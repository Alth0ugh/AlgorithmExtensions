using AlgorithmExtensions.Hyperalgorithms;
using Microsoft.ML;
using Microsoft.ML.Data;
using Microsoft.ML.Trainers;
using Microsoft.ML.Trainers.LightGbm;
using Microsoft.ML.Vision;
using NumSharp.Extensions;
using System.Diagnostics;
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

            pipeline.Add(new Func<LbfgsPoissonRegressionTrainer.Options, LbfgsPoissonRegressionTrainer>(context.Regression.Trainers.LbfgsPoissonRegression), context.Regression.Trainers, "lbfgs");
            pipeline.Add(new Func<AveragedPerceptronTrainer.Options, AveragedPerceptronTrainer>(context.BinaryClassification.Trainers.AveragedPerceptron), context.BinaryClassification.Trainers, "perceptron");
            pipeline.Add(new Func<ImageClassificationTrainer.Options, ImageClassificationTrainer>(context.MulticlassClassification.Trainers.ImageClassification), context.MulticlassClassification.Trainers, "image");

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
                                            path: @"D:\Drive\Škola\UK\2022-2023\Cvièenia\Pokroèilé programovanie v C#\zapoctovy_program\AlgorithmExtensions\AlgorithmExtensions.Tests\creditcard.csv",
                                            hasHeader: true,
                                            separatorChar: ',',
                                            allowQuoting: true,
                                            allowSparse: false);

            var a = mlContext.Data.TrainTestSplit(trainingDataView, 0.2);
            

            var dataProcessPipeline = mlContext.Transforms.Concatenate("Features", new[] { "Time", "V1", "V2", "V3", "V4", "V5", "V6", "V7", "V8", "V9", "V10", "V11", "V12", "V13", "V14", "V15", "V16", "V17", "V18", "V19", "V20", "V21", "V22", "V23", "V24", "V25", "V26", "V27", "V28", "Amount" });

            // Choosing algorithm
            var trainer = mlContext.BinaryClassification.Trainers.LightGbm(numberOfIterations: 100);
            // Appending algorithm to pipeline
            var trainingPipeline = dataProcessPipeline.Append(trainer);

            ITransformer model = trainingPipeline.Fit(a.TrainSet);

            var prediction = model.Transform(trainingDataView);
            var metrics = mlContext.BinaryClassification.EvaluateNonCalibrated(prediction);

            var predictionEngine = mlContext.Model.CreatePredictionEngine<ModelInput, ModelOutput>(model);

            var tt = mlContext.Data.CreateEnumerable<ModelOutput>(prediction, true);
            var count = 0;
            foreach (var k in tt)
            {
                count++;
                if (k.Time == 406)
                {
                    Debug.WriteLine(k);

                }
            }
            
            Debug.WriteLine("");
        }

        [Fact]
        public async Task TestGridSearch()
        {
            var mlContext = new MLContext();

            IDataView trainingDataView = mlContext.Data.LoadFromTextFile<ModelInput>(
                                            path: @"D:\Drive\Škola\UK\2022-2023\Cvièenia\Pokroèilé programovanie v C#\zapoctovy_program\AlgorithmExtensions\AlgorithmExtensions.Tests\creditcard.csv",
                                            hasHeader: true,
                                            separatorChar: ',',
                                            allowQuoting: true,
                                            allowSparse: false);

            var pipelineTamplate = new PipelineTemplate();
            pipelineTamplate.Add(mlContext.Transforms.Concatenate, mlContext.Transforms, "concatenate");
            pipelineTamplate.Add(new Func<LightGbmBinaryTrainer.Options, LightGbmBinaryTrainer>(mlContext.BinaryClassification.Trainers.LightGbm), mlContext.BinaryClassification.Trainers, "lgbm");

            var parameters = new Dictionary<string, string[]>();
            parameters.Add("lgbm", new string[] { "NumberOfIterations__1", "NumberOfIterations__100" });

            var gridSearch = new GridSearchCV<ModelOutput>(mlContext, pipelineTamplate, parameters);
            //await gridSearch.Fit(trainingDataView);
        }
    }
}