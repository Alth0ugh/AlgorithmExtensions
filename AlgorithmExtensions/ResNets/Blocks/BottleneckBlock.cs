﻿using static Tensorflow.Binding;
using Tensorflow;

namespace AlgorithmExtensions.ResNets.Blocks
{
    /// <summary>
    /// Represents bottleneck block in ResNet.
    /// </summary>
    internal class BottleneckBlock : ResidualBlockBase
    {
        /// <inheritdoc/>
        internal override Tensors CreateBlock(Tensors input, int filters, Shape stride, bool useShortcut)
        {
            var shortcut = input;
            if (useShortcut)
            {
                shortcut = tf.keras.layers.Conv2D(filters * 4,
                    new Shape(1, 1),
                    stride,
                    bias_initializer: "zeros").Apply(shortcut);
            }

            var x = tf.keras.layers.Conv2D(filters,
                new Shape(1, 1),
                new Shape(1, 1),
                bias_initializer: "zeros").Apply(input);
            x = tf.keras.layers.BatchNormalization().Apply(x);
            x = tf.keras.layers.LeakyReLU().Apply(x);
            x = tf.keras.layers.Conv2D(filters,
                new Shape(3, 3),
                stride,
                padding: "same",
                bias_initializer: "zeros").Apply(x);
            x = tf.keras.layers.BatchNormalization().Apply(x);
            x = tf.keras.layers.LeakyReLU().Apply(x);
            x = tf.keras.layers.Conv2D(4 * filters,
                new Shape(1, 1),
                new Shape(1 ,1),
                bias_initializer: "zeros").Apply(x);
            x = tf.keras.layers.BatchNormalization().Apply(x);
            x = tf.keras.layers.Add().Apply(new Tensors(x, shortcut));
            x = tf.keras.layers.LeakyReLU().Apply(x);
            return x;
        }
    }
}
