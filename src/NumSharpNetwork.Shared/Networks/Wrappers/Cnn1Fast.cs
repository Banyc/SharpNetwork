using Numpy;
using NumSharpNetwork.Shared.Optimizers;

namespace NumSharpNetwork.Shared.Networks.Wrappers
{
    public class Cnn1Fast : NetworkWrapper
    {
        public override string Name { get; set; } = "CNN-1-Fast";

        public Cnn1Fast(int height, int width, int channels, int numClasses, OptimizerFactory optimizerFactory)
        {
            Convolution2DLayerSet convolution2DLayerSet1 = new Convolution2DLayerSet(channels, 32, optimizerFactory, true, false, name: "Convolution2DLayerSet-1");
            FlattenLayer flatten = new FlattenLayer();

            int linearInputSize = (height / 1) * (width / 1) * 32;

            LinearLayer linear1 = new LinearLayer(optimizerFactory, linearInputSize, numClasses, name: $"{this.Name}.LinearLayer1");

            SoftmaxLayer softmax = new SoftmaxLayer();

            this.Layers.Add(convolution2DLayerSet1);
            this.Layers.Add(flatten);
            this.Layers.Add(linear1);
            this.Layers.Add(softmax);
        }
    }
}
