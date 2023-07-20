using System.Numerics;

namespace AlgorithmExtensions.Hyperalgorithms.ParameterProviders
{
    public class StepParameterProvider<T> : IParameterProvider where T : INumber<T>
    {
        public string Name { get; init; }
        public T Start { get; set; }
        public T Stop { get; set; }
        public T Step { get; set; }

        public StepParameterProvider(string name, T start, T stop, T step)
        {
            Name = name;
            Start = start;
            Stop = stop;
            Step = step;
        }

        public object[] GetParameterValues()
        {
            var current = Start;
            var list = new List<object>() { current };
            do
            {
                list.Add(current);
                current += Step;
            } while (current < Stop);

            return list.ToArray();
        }
    }
}
