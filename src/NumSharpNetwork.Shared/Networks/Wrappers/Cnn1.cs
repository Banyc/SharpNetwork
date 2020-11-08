using Numpy;
using NumSharpNetwork.Shared.Optimizers;

namespace NumSharpNetwork.Shared.Networks.Wrappers
{
    public class Cnn1 : NetworkWrapper
    {
        public override string Name { get; set; } = "CNN-1";

        public Cnn1(int height, int width, int channels, int numClasses, IOptimizer optimizer)
        {
            Convolution2DLayerSet convolution2DLayerSet1 = new Convolution2DLayerSet(channels, 2, optimizer, name: "Convolution2DLayerSet-1");
            FlattenLayer flatten = new FlattenLayer();

            int linearInputSize = (height / 2) * (width / 2) * 2;

            LinearLayer linear1 = new LinearLayer(optimizer, linearInputSize, numClasses, name: $"{this.Name}.LinearLayer1");

            SoftmaxLayer softmax = new SoftmaxLayer();

            this.Layers.Add(convolution2DLayerSet1);
            this.Layers.Add(flatten);
            this.Layers.Add(linear1);
            this.Layers.Add(softmax);
        }
    }
}
