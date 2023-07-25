using AlgorithmExtensions.ResNets.Blocks;
using Tensorflow;

namespace AlgorithmExtensions.Extensions
{
    /// <summary>
    /// Extensions for tensors.
    /// </summary>
    internal static class TensorsExtensions
    {
        /// <summary>
        /// Stacks residual blocks onto input tensors.
        /// </summary>
        /// <param name="input">Input tensors.</param>
        /// <param name="blockBase">Residual block that should be stacked onto the tensors.</param>
        /// <param name="filters">Number of filters to be used in the blocks.</param>
        /// <param name="stride">Stride to be used in the first block of sequence of blocks.</param>
        /// <param name="count">Number of blocks to be stacked.</param>
        /// <returns>Tensors with residual blocks stacked onto them.</returns>
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
