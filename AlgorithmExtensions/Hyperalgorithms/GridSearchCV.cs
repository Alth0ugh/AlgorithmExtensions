﻿using AlgorithmExtensions.Exceptions;
using AlgorithmExtensions.Hyperalgorithms.ParameterProviders;
using AlgorithmExtensions.Scoring;
using Microsoft.ML;
using Microsoft.ML.Data;
using Microsoft.ML.Trainers;

namespace AlgorithmExtensions.Hyperalgorithms
{
    /// <summary>
    /// Class that tests combinations of parameter values of a model, evaluates it and finds the model with the highest score.
    /// </summary>
    public class GridSearchCV
    {
        private PipelineTemplate _template;
        private ParameterProviderForModel _parameters;
        private IScoringFunction _scoringFunction;
        private int _crossValidationSplits;
        private MLContext _mlContext;
        private bool _maximize;
        private bool _multipleThreads;

        private const string _checkParametersParametersNotGivenError = "No parameters were given to {0}";
        private const string _creationalDelegateError = "Creational delegate with name {0} did not create any type of estimator or default parameters given were incorrect.";

        /// <summary>
        /// Estimator trained on the whole dataset using the best parameters.
        /// </summary>
        public ITransformer? BestEstimator { get; set; }
        /// <summary>
        /// Best parameters found.
        /// </summary>
        public Dictionary<string, ParameterInstance[]>? BestParameters { get; set; }

        /// <summary>
        /// Creates new instance of grid search.
        /// </summary>
        /// <param name="mlContext">Machine learning context.</param>
        /// <param name="template">Model template.</param>
        /// <param name="parameters">Parameter provider for the model.</param>
        /// <param name="scoringFunction">Scoring function to be used for scoring the models.</param>
        /// <param name="crossValidationSplits">Number of splits for cross-validation.</param>
        /// <param name="maximize">True if maximizing scoring function, otherwise false.</param>
        public GridSearchCV(MLContext mlContext, PipelineTemplate template, ParameterProviderForModel parameters, IScoringFunction scoringFunction, int crossValidationSplits = 5, bool maximize = true, bool multipleThreads = true)
        {
            _template = template;
            _parameters = parameters;
            _scoringFunction = scoringFunction;
            _mlContext = mlContext;
            _crossValidationSplits = crossValidationSplits;
            _maximize = maximize;
            _multipleThreads = multipleThreads;
        }

        /// <summary>
        /// Fits model with all the combinations of parameters and finds the best one according to scoring function.
        /// </summary>
        /// <param name="data">Data to be trained on.</param>
        /// <returns>Task representing training of the model.</returns>
        /// <exception cref="ParametersMissingException">Thrown when there are no parameters given to GridSearch.</exception>
        /// <exception cref="UniqueValueException">Thrown when there are multiple value providers for a single parameter.</exception>
        public async Task Fit(IDataView data)
        {
            CheckParameters();
            CheckParameterUniqueness();

            if (_multipleThreads)
            {
                await FitMultipleThreads(data);
            }
            else
            {
                FitSingleThread(data);
            }
        }

        private async Task FitMultipleThreads(IDataView data)
        {
            var pipelines = GenerateParameterCombinations().ToArray();
            var tasks = new Task<float>[pipelines.Length];

            for (int i = 0; i < pipelines.Length; i++)
            {
                var pipeline = pipelines[i];
                tasks[i] = Task.Run(() => CrossValidateModel(pipeline, data));
            }

            var results = await Task.WhenAll(tasks);

            var bestValue = _maximize ? results.Max() : results.Min();
            var bestEstimator = 0;
            while (bestValue != results[bestEstimator])
            {
                bestEstimator++;
            }

            BestEstimator = GenerateEstimatorChain(pipelines[bestEstimator]).Fit(data);
            BestParameters = pipelines[bestEstimator];
        }

        private void FitSingleThread(IDataView data)
        {
            var pipelines = GenerateParameterCombinations().ToArray();
            var results = new float[pipelines.Length];

            for (int i = 0; i < pipelines.Length; i++)
            {
                var pipeline = pipelines[i];
                results[i] = CrossValidateModel(pipeline, data);
            }

            var bestValue = _maximize ? results.Max() : results.Min();
            var bestEstimator = 0;
            while (bestValue != results[bestEstimator])
            {
                bestEstimator++;
            }

            BestEstimator = GenerateEstimatorChain(pipelines[bestEstimator]).Fit(data);
            BestParameters = pipelines[bestEstimator];
        }

        /// <summary>
        /// Generates all combinations of parameter values for a model.
        /// </summary>
        /// <returns>Enumerable containing the parameter values combinations.</returns>
        private IEnumerable<Dictionary<string, ParameterInstance[]>> GenerateParameterCombinations()
        {
            var domains = new List<int>();
            var estimatorNames = _parameters.Keys.ToArray();
            foreach (var key in estimatorNames)
            {
                var modelParameters = _parameters[key];
                foreach (var parameter in modelParameters)
                {
                    domains.Add(parameter.GetParameterValues().Length);
                }
            }

            var indexCombinations = GenerateIndexCombinations(domains.ToArray());

            foreach (var combination in indexCombinations)
            {
                var dict = new Dictionary<string, ParameterInstance[]>();
                var index = 0;
                for (int i = 0; i < estimatorNames.Length; i++)
                {
                    var parameterInstances = new List<ParameterInstance>();
                    for (int j = 0; j < _parameters[estimatorNames[i]].Length; j++)
                    {
                        var possibleParameters = _parameters[estimatorNames[i]][j].GetParameterValues();
                        parameterInstances.Add(new ParameterInstance(_parameters[estimatorNames[i]][j].Name, possibleParameters[combination[index]]));
                        index++;
                    }
                    dict.Add(estimatorNames[i], parameterInstances.ToArray());
                }
                yield return dict;
            }
        }

        /// <summary>
        /// Generates combinations of indices for a list of lists. From each list is picked exactly one index 
        /// and the combinations of such indices are generated.
        /// </summary>
        /// <param name="domain">Number of elements in each list.</param>
        /// <returns>Enumerable of index combinations.</returns>
        private IEnumerable<List<int>> GenerateIndexCombinations(int[] domain)
        {
            if (domain.Length == 0)
            {
                yield break;
            }
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

        private void CheckParameters()
        {
            if (_parameters.Count == 0)
            {
                throw new ParametersMissingException(string.Format(_checkParametersParametersNotGivenError, nameof(GridSearchCV)));
            }

            var correctValues = from val in _parameters
                                where val.Value.Length > 0
                                select val;
            if (correctValues.Count() == 0)
            {
                throw new ParametersMissingException(string.Format(_checkParametersParametersNotGivenError, nameof(GridSearchCV)));
            }
        }

        private void CheckParameterUniqueness()
        {
            foreach (var item in _parameters)
            {
                var uniqueValues = item.Value.DistinctBy(provider => provider.Name);
                if (uniqueValues.Count() != item.Value.Length)
                {
                    throw new UniqueValueException($"There are multiple providers for one parameter for {item.Value}");
                }
            }
        }

        /// <summary>
        /// Cross-validates a model.
        /// </summary>
        /// <param name="parameters">Parameters to be used for training.</param>
        /// <param name="data">Data to be trained on.</param>
        /// <returns>Score computed be averaging the individual scores of each fold.</returns>
        private float CrossValidateModel(Dictionary<string, ParameterInstance[]> parameters, IDataView data)
        {
            var splits = _mlContext.Data.CrossValidationSplit(data, _crossValidationSplits);
            var results = new List<float>();
            foreach (var split in splits)
            {
                IEstimator<ITransformer> instance = GenerateEstimatorChain(parameters);
                var trainedInstance = instance.Fit(split.TrainSet);
                var transformedData = trainedInstance.Transform(split.TestSet);
                results.Add(_scoringFunction.Score(transformedData));
            }

            var sum = results.Sum();
            return sum / results.Count;
        }

        /// <summary>
        /// Generates model using a set of parameter values.
        /// </summary>
        /// <param name="parameterCollection">Parameters to be used for the model.</param>
        /// <returns>Generated model with given parameters.</returns>
        private EstimatorChain<ITransformer> GenerateEstimatorChain(Dictionary<string, ParameterInstance[]> parameterCollection)
        {
            var estimator = new EstimatorChain<ITransformer>();
            foreach (var item in _template.Items)
            {
                estimator = estimator.Append(GenerateInstanceFromParameters(item, parameterCollection));
            }
            return estimator;
        }

        /// <summary>
        /// Generates an instance of estimator or transformer with given parameters.
        /// </summary>
        /// <param name="pipelineItem">Item to generate instance of.</param>
        /// <param name="parameterCollection">Set of parameters for the model.</param>
        /// <returns>Instance of estimator or transformer.</returns>
        /// <exception cref="IncorrectCreationalDelegateException">Thrown if the creational delegate is in incorrect format (function with input parameter of type TrainerInputBase)</exception>
        private IEstimator<ITransformer> GenerateInstanceFromParameters(PipelineItem pipelineItem, Dictionary<string, ParameterInstance[]> parameterCollection)
        {
            var creationalDelegate = pipelineItem.CreationalDelegate;
            var delegateParameters = creationalDelegate.Method.GetParameters();
            var optionTypes = from parameter in delegateParameters 
                              where parameter.ParameterType.IsSubclassOf(typeof(TrainerInputBase)) || 
                                typeof(IOptions).IsAssignableFrom(parameter.ParameterType)
                              select parameter;

            var optionTypesCount = optionTypes.Count();
            if (optionTypesCount > 1)
            {
                throw new IncorrectCreationalDelegateException(optionTypesCount > 1 ? "There are multiple option parameters in creational delegate" : "There is no option parameter in creational delegate");
            }

            IEstimator<ITransformer>? estimator = null;

            if (optionTypesCount == 1)
            {
                var options = GenerateAndSetOptions(optionTypes.First().ParameterType, pipelineItem, parameterCollection);
                estimator = MakeInstanceOfEstimator(creationalDelegate, options);
            }
            else
            {
                try
                {
                    var parameters = AppendDefaultDelegateParameters(creationalDelegate, pipelineItem.DefaultParameters);
                    estimator = (IEstimator<ITransformer>)creationalDelegate.DynamicInvoke(parameters)!;
                }
                catch
                {
                    throw new IncorrectCreationalDelegateException(string.Format(_creationalDelegateError, pipelineItem.Name));
                }
            }

            if (estimator is null)
            {
                throw new IncorrectCreationalDelegateException(string.Format(_creationalDelegateError, pipelineItem.Name));
            }

            return estimator;
        }


        private object[] AppendDefaultDelegateParameters(Delegate creationalDelegate, object[]? parametersGiven)
        {
            var delegateParameters = creationalDelegate.Method.GetParameters();
            var allParameters = new List<object>(parametersGiven ?? new object[0]);
            var offset = creationalDelegate.Target is null ? 0 : 1;

            for (var i = parametersGiven?.Length + offset ?? 0 + offset; i < delegateParameters.Length; i++)
            {
                allParameters.Add(delegateParameters[i].DefaultValue!);
            }
            return allParameters.ToArray();
        }

        /// <summary>
        /// Generates options for model and sets options according to parameters set by the user values and deafult values.
        /// </summary>
        /// <param name="optionType">Type of the options to be generated.</param>
        /// <param name="pipelineItem">Item that options are generated for.</param>
        /// <param name="parameterCollection">Parameters for model.</param>
        /// <returns>Options with set parameter values.</returns>
        private object GenerateAndSetOptions(Type optionType, PipelineItem pipelineItem, Dictionary<string, ParameterInstance[]> parameterCollection)
        {
            var options = Activator.CreateInstance(optionType)!;
            var defaultOptions = pipelineItem.DefaultOptions;

            if (defaultOptions != null)
            {
                SetDefaultOptions(options, defaultOptions);
            }

            if (parameterCollection.ContainsKey(pipelineItem.Name))
            {
                foreach (var parameter in parameterCollection[pipelineItem.Name])
                {
                    SetPropertyOfOptions(options, parameter.Name, parameter.Value);
                }
            }

            return options;
        }

        /// <summary>
        /// Sets default parameter values in options.
        /// </summary>
        /// <param name="options">Options to set default values of.</param>
        /// <param name="defaultOptions">Options with default values set.</param>
        /// <exception cref="OptionTypeMismatchException">Thrown if options and defaultOptions do not match in type.</exception>
        private void SetDefaultOptions(object options, object defaultOptions)
        {
            if (options.GetType() != defaultOptions.GetType())
            {
                throw new OptionTypeMismatchException("Model option type and supplied default option type does not match", options.GetType(), defaultOptions.GetType());
            }

            var properties = options.GetType().GetFields();
            foreach (var property in properties)
            {
                property.SetValue(options, property.GetValue(defaultOptions));
            }
        }

        /// <summary>
        /// Instantiates an estimator with options.
        /// </summary>
        /// <param name="creationalDelegate">Delegate that instantiates the estimator.</param>
        /// <param name="options">Options to be used for instantiation.</param>
        /// <returns>Instance of the estimator.</returns>
        /// <exception cref="IncorrectCreationalDelegateException">Thrown if the creational delegate is in incorrect format (function with input parameter of type TrainerInputBase)</exception>
        private IEstimator<ITransformer> MakeInstanceOfEstimator(Delegate creationalDelegate, object options)
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

        /// <summary>
        /// Sets a property of options to a value.
        /// </summary>
        /// <param name="options">Options to set value of a property.</param>
        /// <param name="parameterName">Name of the property to be set.</param>
        /// <param name="parameterValue">Value of the property.</param>
        /// <exception cref="IncorrectOptionParameterException">Thrown if the property is not found.</exception>
        private void SetPropertyOfOptions(object options, string parameterName, object parameterValue)
        {
            var property = from prop in options.GetType().GetFields()
                           where prop.Name == parameterName
                           select prop;

            if (property.Count() == 0)
            {
                throw new IncorrectOptionParameterException($"Parameter {parameterName} was not found on object of type {options.GetType().FullName}");
            }

            property.First().SetValue(options, parameterValue);
        }
    }
}
