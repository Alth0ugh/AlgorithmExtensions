using Tensorflow;

namespace AlgorithmExtensions.ResNets
{
    public enum ResNetArchitecture
    {
        ResNet18,
        ResNet34,
        ResNet50,
        ResNet101,
        ResNet152
    }
    public class Options
    {
        public ResNetArchitecture Architecture { get; set; }
        public Shape InputShape { get; set; }
        public int Classes { get; set; }
    }
}