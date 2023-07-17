using AlgorithmExtensions.Exceptions;
using AlgorithmExtensions.Scoring;
using Microsoft.ML;
using Microsoft.ML.Data;
using Microsoft.ML.Trainers;
using Newtonsoft.Json.Serialization;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using static Microsoft.ML.TrainCatalogBase;
using static System.Runtime.InteropServices.JavaScript.JSType;

namespace AlgorithmExtensions.Hyperalgorithms
{
    public class GridSearchCV<TOut>
    {
        private PipelineTemplate _template;
        private Dictionary<string, string[]> _parameters;
        private IScoringFunction _scoringFunction;
        private readonly int _numberOfJobs;
        private bool _refit;
        private int _crossValidationSplits;
        private MLContext _mlContext;
        private IEstimator<ITransformer>[] _estimators = null;
        public IEstimator<ITransformer> BestEstimator { get; set; }

        public GridSearchCV(MLContext mlContext, PipelineTemplate template, Dictionary<string, string[]> parameters, IScoringFunction scoringFunction = null, int numberOfJobs = -1, bool refit = true, int crossValidationSplits = 5)
        {
            _template = template;
            _parameters = parameters;
            _scoringFunction = scoringFunction;
            _numberOfJobs = numberOfJobs;
            _refit = refit;
            _mlContext = mlContext;
            _crossValidationSplits = crossValidationSplits;
        }

        public IEnumerable<Dictionary<string, string>> GenerateParameterCombinations()
        {
            var kvp = _parameters.ToArray();
            var domains = new List<int>();
            foreach (var kvp2 in kvp)
            {
                domains.Add(kvp2.Value.Length);
            }

            var indexCombinations = GenerateIndexCombinations(domains.ToArray());

            foreach (var combination in indexCombinations)
            {
                var dict = new Dictionary<string, string>();

                for (int i = 0; i < kvp.Length; i++)
                {
                    dict.Add(kvp[i].Key, kvp[i].Value[combination[i]]);
                }

                yield return dict;
            }
        }

        public IEnumerable<List<int>> GenerateIndexCombinations(int[] domain)
        {
            if (domain.Length == 1)
            {
                for (int i = 0; i < domain[0]; i++)
                {
                    yield return new List<int>() { i };
                }
                yield break;
            }

            var previous = GenerateIndexCombinations(domain[1..]);
            for (int i = 0; i < domain[0]; i++)
            {
                foreach (var subdomain in previous)
                {
                    yield return subdomain.Prepend(i).ToList();
                }
            }
        }

        public IEnumerable<IEstimator<ITransformer>> GeneratePipelinesFromParameters()
        {
            var parameterDictionaries = GenerateParameterCombinations();
            foreach (var parameterCollection in parameterDictionaries)
            {
                var estimator = GenerateEstimatorChain(parameterCollection);
                yield return estimator;
            }
        }

        public async Task Fit(IDataView data)
        {
            var pipelines = GenerateParameterCombinations().ToArray();
            var tasks = new Task<float>[pipelines.Length];

            for (int i = 0; i <= pipelines.Length; i++)
            {
                tasks[i] = Task.Run(() => CrossValidateModel(pipelines[i], data));
            }

            var results = await Task.WhenAll(tasks);

            var maxValue = results.Max();
            var bestEstimator = 0;
            while (maxValue != results[bestEstimator])
            {
                bestEstimator++;
            }

            BestEstimator = GenerateEstimatorChain(pipelines[bestEstimator]);
        }

        private float CrossValidateModel(Dictionary<string, string> parameters, IDataView data)
        {
            var splits = _mlContext.Data.CrossValidationSplit(data, _crossValidationSplits);
            var results = new List<float>();
            foreach (var split in splits)
            {
                IEstimator<ITransformer> instance = GenerateEstimatorChain(parameters);
                var trainedInstance = instance.Fit(split.TrainSet);
                var transformedData = trainedInstance.Transform(split.TestSet);
                results.Add(_scoringFunction.Score(split.TrainSet, transformedData));
            }

            var sum = results.Sum();
            return sum / results.Count;
        }


        private EstimatorChain<ITransformer> GenerateEstimatorChain(Dictionary<string, string> parameterCollection)
        {
            var estimator = new EstimatorChain<ITransformer>();
            foreach (var item in _template.Delegates)
            {
                estimator.Append(GenerateInstanceFromParameters(item, parameterCollection));
            }
            return estimator;
        }

        private IEstimator<ITransformer> GenerateInstanceFromParameters(PipelineItem pipelineItem, Dictionary<string, string> parameterCollection)
        {
            var creationalDelegate = pipelineItem.CreationalDelegate;
            var delegateParameters = creationalDelegate.Method.GetParameters();
            var optionTypes = from parameter in delegateParameters 
                              where parameter.ParameterType.IsSubclassOf(typeof(TrainerInputBase)) 
                              select parameter;

            var optionTypesCount = optionTypes.Count();
            if (optionTypesCount > 1 && optionTypesCount < 1)
            {
                throw new IncorrectCreationalDelegateException(optionTypesCount > 1 ? "There are multiple option parameters in creational delegate" : "There is no option parameter in creational delegate");
            }

            var optionsType = optionTypes.First().ParameterType;
            var options = (TrainerInputBase)Activator.CreateInstance(optionsType);

            if (parameterCollection.ContainsKey(pipelineItem.Name))
            {
                var parameterEntry = parameterCollection[pipelineItem.Name].Split("_");
                if (parameterEntry.Length > 2)
                {
                    throw new IncorrectParameterFormatException($"Input parameter {parameterCollection[pipelineItem.Name]} is in incorrect format.");
                }
                SetPropertyOfOptions(options, parameterEntry[0], parameterEntry[1]);
            }

            return MakeInstanceOfEstimator(creationalDelegate, options);
        }

        private IEstimator<ITransformer> MakeInstanceOfEstimator(Delegate creationalDelegate, TrainerInputBase options)
        {
            try
            {
                return (IEstimator<ITransformer>)creationalDelegate.DynamicInvoke(options)!;
            }
            catch (InvalidCastException)
            {
                throw new IncorrectCreationalDelegateException("Creational delegate given did not create any type of estimator.");
            }
            catch
            {
                throw;
            }
        }

        private void SetPropertyOfOptions(TrainerInputBase options, string parameterName, string parameterValue)
        {
            var property = from prop in options.GetType().GetFields()
                           where prop.Name == parameterName
                           select prop;

            if (property.Count() == 0)
            {
                throw new IncorrectOptionParameterException($"Parameter {parameterName} was not found on object of type {options.GetType().FullName}");
            }

            var setType = property.First().FieldType;
            object convertedParameter = null;
            
            try
            {
                convertedParameter = Convert.ChangeType(parameterValue, setType);
            }
            catch
            {
                throw new ParameterConversionException($"Parameter {parameterName} could not be converted to type {setType.Name}");
            }

            property.First().SetValue(options, convertedParameter);
        }
    }
}
