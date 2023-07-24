using AlgorithmExtensions.ResNets.Blocks;
using Tensorflow;

namespace AlgorithmExtensions.Extensions
{
    internal static class TensorsExtensions
    {
        internal static Tensors StackResidualBlocks(this Tensors input, ResidualBlockBase blockBase, int filters, Shape stride, int count)
        {
            var block1 = blockBase.CreateBlock(input, filters, stride, true);
            for (int i = 1; i < count; i++)
            {
                block1 = blockBase.CreateBlock(block1, filters, new Shape(1, 1), false);
            }
            return block1;
        }
    }
}
