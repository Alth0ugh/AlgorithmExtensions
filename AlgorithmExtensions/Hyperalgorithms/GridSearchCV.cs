using AlgorithmExtensions.Exceptions;
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
        private bool _refit;
        private int _crossValidationSplits;
        private MLContext _mlContext;
        private IEstimator<ITransformer>[] _estimators = null;
        public IEstimator<ITransformer> BestEstimator { get; set; }
        public ParameterProviderForModel BestParameters { get; set; }

        /// <summary>
        /// Creates new instance of grid search.
        /// </summary>
        /// <param name="mlContext">Machine learning context.</param>
        /// <param name="template">Model template.</param>
        /// <param name="parameters">Parameter provider for the model.</param>
        /// <param name="scoringFunction">Scoring function to be used for scoring the models.</param>
        /// <param name="refit">If true, the best model is refitted on the whole dataset.</param>
        /// <param name="crossValidationSplits">Number of splits for cross-validation.</param>
        public GridSearchCV(MLContext mlContext, PipelineTemplate template, ParameterProviderForModel parameters, IScoringFunction scoringFunction = null, bool refit = true, int crossValidationSplits = 5)
        {
            _template = template;
            _parameters = parameters;
            _scoringFunction = scoringFunction;
            _refit = refit;
            _mlContext = mlContext;
            _crossValidationSplits = crossValidationSplits;
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
                        parameterInstances.Add(new ParameterInstance() 
                        { 
                            Name = _parameters[estimatorNames[i]][j].Name, 
                            Value = possibleParameters[combination[index]] 
                        });
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

        /// <summary>
        /// Fits model with all the combinations of parameters and finds the best one according to scoring function.
        /// </summary>
        /// <param name="data">Data to be trained on.</param>
        /// <returns>Task representing training of the model.</returns>
        public async Task Fit(IDataView data)
        {
            var pipelines = GenerateParameterCombinations().ToArray();
            var tasks = new Task<float>[pipelines.Length];

            for (int i = 0; i < pipelines.Length; i++)
            {
                var pipeline = pipelines[i];
                tasks[i] = Task.Run(() => CrossValidateModel(pipeline, data));
            }

            var results = await Task.WhenAll(tasks);

            var maxValue = results.Max();
            var bestEstimator = 0;
            while (maxValue != results[bestEstimator])
            {
                bestEstimator++;
            }

            BestEstimator = GenerateEstimatorChain(pipelines[bestEstimator]);
            //BestParameters = pipelines[bestEstimator];
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
                              where parameter.ParameterType.IsSubclassOf(typeof(TrainerInputBase)) 
                              select parameter;

            var optionTypesCount = optionTypes.Count();
            if (optionTypesCount > 1 && optionTypesCount < 1)
            {
                throw new IncorrectCreationalDelegateException(optionTypesCount > 1 ? "There are multiple option parameters in creational delegate" : "There is no option parameter in creational delegate");
            }

            IEstimator<ITransformer> estimator = null;

            if (optionTypesCount == 1)
            {
                var options = GenerateAndSetOptions(optionTypes.First().ParameterType, pipelineItem, parameterCollection);
                estimator = MakeInstanceOfEstimator(creationalDelegate, options);
            }
            else
            {
                try
                {
                    estimator = (IEstimator<ITransformer>)creationalDelegate.DynamicInvoke(pipelineItem.DefaultParameters);
                }
                catch
                {
                    throw new IncorrectCreationalDelegateException($"Creational delegate with name {pipelineItem.Name} did not create any type of estimator or default parameters given were incorrect.");
                }
            }
            return estimator;
        }

        /*
        private IEstimator<ITransformer> CreateTransformerFromDelegate(PipelineItem pipelineItem, Dictionary<string, string> parameterCollection)
        {
            var defaultParameters = pipelineItem.DefaultParameters;
            var delegateParameters = pipelineItem.CreationalDelegate.Method.GetParameters();
            var parameterEntry = parameterCollection[pipelineItem.Name].Split("_");
            var targetParameterName = parameterEntry[0];
            var targetParameterValue = parameterEntry[1];

            var targetParameter = from parameter in delegateParameters
                                  where parameter.Name == targetParameterName
                                  select parameter;

            return null;
        }
        */

        /// <summary>
        /// Generates options for model and sets options according to parameters set by the user values and deafult values.
        /// </summary>
        /// <param name="optionType">Type of the options to be generated.</param>
        /// <param name="pipelineItem">Item that options are generated for.</param>
        /// <param name="parameterCollection">Parameters for model.</param>
        /// <returns>Options with set parameter values.</returns>
        private TrainerInputBase GenerateAndSetOptions(Type optionType, PipelineItem pipelineItem, Dictionary<string, ParameterInstance[]> parameterCollection)
        {
            var options = (TrainerInputBase)Activator.CreateInstance(optionType);
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
        private void SetDefaultOptions(TrainerInputBase options, TrainerInputBase defaultOptions)
        {
            if (options.GetType() != defaultOptions.GetType())
            {
                throw new OptionTypeMismatchException("Model option type and supplied default option type does not match", options.GetType(), defaultOptions.GetType());
            }

            var properties = options.GetType().GetProperties();
            foreach (var property in properties )
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

        /// <summary>
        /// Sets a property of options to a value.
        /// </summary>
        /// <param name="options">Options to set value of a property.</param>
        /// <param name="parameterName">Name of the property to be set.</param>
        /// <param name="parameterValue">Value of the property.</param>
        /// <exception cref="IncorrectOptionParameterException">Thrown if the property is not found.</exception>
        private void SetPropertyOfOptions(TrainerInputBase options, string parameterName, object parameterValue)
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
