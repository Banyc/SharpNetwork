using Numpy;
using NumSharpNetwork.Shared.Optimizers;

namespace NumSharpNetwork.Shared.Networks.Wrappers
{
    public class Cnn : NetworkWrapper
    {
        public override string Name { get; set; } = "CNN";

        public Cnn(int height, int width, int channels, int numClasses, IOptimizer optimizer)
        {
            Convolution2DLayerSet convolution2DLayerSet1 = new Convolution2DLayerSet(channels, 32, optimizer, name: "Convolution2DLayerSet-1");
            Convolution2DLayerSet convolution2DLayerSet2 = new Convolution2DLayerSet(32, 64, optimizer, name: "Convolution2DLayerSet-2");
            FlattenLayer flatten = new FlattenLayer();

            int linearInputSize = (height / 4) * (width / 4) * 64;

            LinearLayer linear1 = new LinearLayer(optimizer, linearInputSize, numClasses);

            SoftmaxLayer softmax = new SoftmaxLayer();

            this.Layers.Add(convolution2DLayerSet1);
            this.Layers.Add(convolution2DLayerSet2);
            this.Layers.Add(flatten);
            this.Layers.Add(linear1);
            this.Layers.Add(softmax);
        }
    }
}
