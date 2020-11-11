using NumSharpNetwork.Shared.Optimizers;

namespace NumSharpNetwork.Shared.Networks.Wrappers
{
    public class ImageLinearLayers : NetworkWrapper
    {
        public override string Name { get; set; } = "ImageLinearLayers";

        public ImageLinearLayers(int height, int width, int channels, int numClasses, OptimizerFactory optimizerFactory)
        {
            FlattenLayer flatten = new FlattenLayer();

            int linearInputSize = (height) * (width) * channels;

            LinearLayer linear1 = new LinearLayer(optimizerFactory, linearInputSize, 100, name: $"{this.Name}.LinearLayer1");

            ReLULayer reLU1 = new ReLULayer();

            LinearLayer linear2 = new LinearLayer(optimizerFactory, 100, numClasses, name: $"{this.Name}.LinearLayer2");

            SoftmaxLayer softmax = new SoftmaxLayer();

            this.Layers.Add(flatten);
            this.Layers.Add(linear1);
            this.Layers.Add(reLU1);
            this.Layers.Add(linear2);
            this.Layers.Add(softmax);
        }
    }
}
