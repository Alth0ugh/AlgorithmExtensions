using System.Numerics;

namespace AlgorithmExtensions.Hyperalgorithms.ParameterProviders
{
    /// <summary>
    /// Provides values from a given interval with given steps.
    /// </summary>
    /// <typeparam name="T">Type of the parameter that is being provided.</typeparam>
    public class StepParameterProvider<T> : IParameterProvider where T : INumber<T>
    {
        /// <inheritdoc/>
        public string Name { get; init; }
        /// <summary>
        /// Starting value.
        /// </summary>
        public T Start { get; set; }
        /// <summary>
        /// When stop value is reached, the stepping is stopped. Stop value is included in the output.
        /// </summary>
        public T Stop { get; set; }
        /// <summary>
        /// Length of the step to take.
        /// </summary>
        public T Step { get; set; }

        /// <summary>
        /// Creates new instance of the provider.
        /// </summary>
        /// <param name="name">Name of the parameter.</param>
        /// <param name="start">Starting value.</param>
        /// <param name="stop">Stop value.</param>
        /// <param name="step">Step length (positive).</param>
        /// <exception cref="ArgumentException">Thrown if the step provider parameter will cause infinite parameter array.</exception>
        public StepParameterProvider(string name, T start, T stop, T step)
        {
            Name = name;
            Start = start;
            Stop = stop;
            Step = step;

            if (Stop < Start)
            {
                throw new ArgumentException($"'{stop}' has to be greater or equal to '{start}'");
            }
            else if (T.IsZero(Step) || T.IsNegative(Step)) 
            {
                throw new ArgumentException($"{step} must be a positive number");
            }
        }

        /// <inheritdoc/>
        public object[] GetParameterValues()
        {
            var current = Start;
            var list = new List<object>() { current };

            while (current <= Stop)
            {
                list.Add(current);
                current += Step;
            }

            return list.ToArray();
        }
    }
}
