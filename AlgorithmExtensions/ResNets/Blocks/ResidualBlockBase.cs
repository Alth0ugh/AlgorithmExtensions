using Tensorflow;

namespace AlgorithmExtensions.ResNets.Blocks
{
    internal abstract class ResidualBlockBase
    {
        internal abstract Tensors CreateBlock(Tensors input, int filters, Shape stride, bool useShortcut);
    }
}
