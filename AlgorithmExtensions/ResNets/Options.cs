using AlgorithmExtensions.Hyperalgorithms;
using Microsoft.ML.Trainers;

namespace AlgorithmExtensions.ResNets
{
    /// <summary>
    /// ResNet architecture enum.
    /// </summary>
    public enum ResNetArchitecture
    {
        ResNet18,
        ResNet34,
        ResNet50,
        ResNet101,
        ResNet152
    }

    /// <summary>
    /// Options for ResNet.
    /// </summary>
    public class Options : IOptions
    {
        public ResNetArchitecture Architecture = ResNetArchitecture.ResNet50;
        public int Classes = 10;
        public string LabelColumnName = "Label";
        public string FeatureColumnName = "Features";
        public int BatchSize = -1;
        public int Epochs = 10;
        public float LearningRate = 0.0001f;
        public int Height = 224;
        public int Width = 224;
    }
}