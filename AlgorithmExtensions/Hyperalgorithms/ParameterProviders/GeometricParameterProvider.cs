using System.Numerics;

namespace AlgorithmExtensions.Hyperalgorithms.ParameterProviders
{
    public class GeometricParameterProvider<T> : IParameterProvider where T : INumber<T>
    {
        public string Name { get; set; }

        private T _start;
        private T _factor;
        private int _count;

        public GeometricParameterProvider(string name, T start, int count, T factor)
        {
            Name = name;
            _start = start;
            _count = count;
            _factor = factor;
        }

        public object[] GetParameterValues()
        {
            if (_count == 0)
            {
                return new object[0];
            }

            var result = new object[_count];
            var lastResult = _start;
            result[0] = _start;

            for (int i = 1; i < _count; i++)
            {
                lastResult *= _factor;
                result[i] = lastResult;
            }

            return result;
        }
    }
}
