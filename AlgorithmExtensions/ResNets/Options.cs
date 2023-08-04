using Tensorflow;

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
    public class Options
    {
        public ResNetArchitecture Architecture { get; set; } = ResNetArchitecture.ResNet50;
        public Shape InputShape { get; set; }
        public int Classes { get; set; }
        public string LabelColumnName { get; set; } = "Label";
        public string PredictedLabelColumnName { get; set; } = "Prediction";
        public string FeatureColumnName { get; set; } = "Features";
        public int BatchSize { get; set; }
        public int Epochs { get; set; }
        public float LearningRate { get; set; } = 0.0001f;
    }
}