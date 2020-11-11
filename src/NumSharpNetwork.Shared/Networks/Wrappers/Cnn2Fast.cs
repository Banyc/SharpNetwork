using Numpy;
using NumSharpNetwork.Shared.Optimizers;

namespace NumSharpNetwork.Shared.Networks.Wrappers
{
    public class Cnn2Fast : NetworkWrapper
    {
        public override string Name { get; set; } = "CNN-2-Fast";

        public Cnn2Fast(int height, int width, int channels, int numClasses, OptimizerFactory optimizerFactory)
        {
            Convolution2DLayerSet convolution2DLayerSet1 = new Convolution2DLayerSet(channels, 32, optimizerFactory, true, false, name: $"{this.Name}.Convolution2DLayerSet-1");
            Convolution2DLayerSet convolution2DLayerSet2 = new Convolution2DLayerSet(32, 64, optimizerFactory, true, false, name: $"{this.Name}.Convolution2DLayerSet-2");
            FlattenLayer flatten = new FlattenLayer();

            int linearInputSize = (height / 1) * (width / 1) * 64;

            LinearLayer linear1 = new LinearLayer(optimizerFactory, linearInputSize, numClasses, name: $"{this.Name}.LinearLayer1");

            SoftmaxLayer softmax = new SoftmaxLayer();

            this.Layers.Add(convolution2DLayerSet1);
            this.Layers.Add(convolution2DLayerSet2);
            this.Layers.Add(flatten);
            this.Layers.Add(linear1);
            this.Layers.Add(softmax);
        }
    }
}
