using Numpy;
using NumSharpNetwork.Shared.Optimizers;

namespace NumSharpNetwork.Shared.Networks.Wrappers
{
    public class Cnn2 : NetworkWrapper
    {
        public override string Name { get; set; } = "CNN-2";

        public Cnn2(int height, int width, int channels, int numClasses, IOptimizer optimizer)
        {
            Convolution2DLayerSet convolution2DLayerSet1 = new Convolution2DLayerSet(channels, 32, optimizer, name: $"{this.Name}.Convolution2DLayerSet-1");
            Convolution2DLayerSet convolution2DLayerSet2 = new Convolution2DLayerSet(32, 64, optimizer, name: $"{this.Name}.Convolution2DLayerSet-2");
            FlattenLayer flatten = new FlattenLayer();

            int linearInputSize = (height / 4) * (width / 4) * 64;

            LinearLayer linear1 = new LinearLayer(optimizer, linearInputSize, numClasses, name: $"{this.Name}.LinearLayer1");

            SoftmaxLayer softmax = new SoftmaxLayer();

            this.Layers.Add(convolution2DLayerSet1);
            this.Layers.Add(convolution2DLayerSet2);
            this.Layers.Add(flatten);
            this.Layers.Add(linear1);
            this.Layers.Add(softmax);
        }
    }
}
