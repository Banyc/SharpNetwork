using NumSharpNetwork.Shared.Optimizers;

namespace NumSharpNetwork.Shared.Networks.Wrappers
{
    public class Convolution2DLayerSet : NetworkWrapper
    {
        public Convolution2DLayerSet(int inputChannels, int outputChannels, IOptimizer optimizer, bool isIm2Col, bool hasPooling, string name = "Convolution2DLayerSet")
        {
            this.Name = name;

            Convolution2D convolution2D = new Convolution2D(inputChannels, outputChannels, optimizer, name: $"{this.Name}.Convolution2D")
            {
                IsIm2Col = isIm2Col
            };
            ReLULayer reLU = new ReLULayer();

            this.Layers.Add(convolution2D);
            this.Layers.Add(reLU);
            if (hasPooling)
            {
                MaxPool2DLayer maxPool2D = new MaxPool2DLayer(2, 2);
                this.Layers.Add(maxPool2D);
            }
        }
    }
}
