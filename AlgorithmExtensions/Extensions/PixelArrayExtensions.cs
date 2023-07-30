using AlgorithmExtensions.ResNets;

namespace AlgorithmExtensions.Extensions
{
    internal static class PixelArrayExtensions
    {
        internal static byte[,,] ToByteArray(this Pixel[,] pixels)
        {
            var result = new byte[pixels.GetLength(0), pixels.GetLength(1), 3];

            for (int i = 0; i < pixels.GetLength(0); i++)
            {
                for (int j = 0; j < pixels.GetLength(1); j++)
                {
                    result[i, j, 0] = pixels[i, j].R;
                    result[i,j, 1] = pixels[i,j].G;
                    result[i,j, 2]  = pixels[i,j].B;
                }
            }
            return result;
        }
    }
}
