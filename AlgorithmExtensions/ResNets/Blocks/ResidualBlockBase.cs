using Tensorflow;

namespace AlgorithmExtensions.ResNets.Blocks
{
    internal abstract class ResidualBlockBase
    {
        /// <summary>
        /// Creates the residual block and applies it to input tensors.
        /// </summary>
        /// <param name="input">Input tensors.</param>
        /// <param name="filters">Number of filters to be used in the block.</param>
        /// <param name="stride">Stride to be used in the block.</param>
        /// <param name="useShortcut">True if shortcut connection should be used, otherwise false.</param>
        /// <returns>Tensors with the block applied to them.</returns>
        internal abstract Tensors CreateBlock(Tensors input, int filters, Shape stride, bool useShortcut);
    }
}
