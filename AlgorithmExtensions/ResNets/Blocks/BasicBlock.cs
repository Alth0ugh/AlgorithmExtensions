﻿using static Tensorflow.Binding;
using Tensorflow;

namespace AlgorithmExtensions.ResNets.Blocks
{
    /// <summary>
    /// Represents basic block in ResNet.
    /// </summary>
    internal class BasicBlock : ResidualBlockBase
    {
        /// <inheritdoc/>
        internal override Tensors CreateBlock(Tensors input, int filters, Shape stride, bool useShortcut)
        {
            var shortcut = input;
            if (useShortcut)
            {
                shortcut = tf.keras.layers.Conv2D(filters * 2,
                    new Shape(1, 1),
                    stride,
                    bias_initializer: "zeros").Apply(shortcut);
            }

            var x = tf.keras.layers.Conv2D(filters,
                kernel_size: new Shape(3, 3),
                strides: stride,
                bias_initializer: "zeros",
                padding: "same").Apply(input);
            x = tf.keras.layers.BatchNormalization().Apply(x);
            x = tf.keras.layers.LeakyReLU().Apply(x);
            x = tf.keras.layers.Conv2D(filters * 2,
                kernel_size: new Shape(3, 3),
                strides: new Shape(1, 1),
                bias_initializer: "zeros",
                padding: "same").Apply(x);
            x = tf.keras.layers.BatchNormalization().Apply(x);

            var add = tf.keras.layers.Add().Apply(new Tensors(x, shortcut));
            var act2 = tf.keras.layers.LeakyReLU().Apply(add);

            return act2;
        }
    }
}
