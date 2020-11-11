using Numpy;
using NumSharpNetwork.Shared.Optimizers;

namespace NumSharpNetwork.Shared.Networks.Wrappers
{
    public class ThreeLinearLayers : NetworkWrapper
    {
        public override string Name { get; set; } = "three-linear-layers";

        public ThreeLinearLayers(OptimizerFactory optimizerFactory)
        {
            LinearLayer linearLayer1 = new LinearLayer(optimizerFactory, 10, 1000)
            {
                Name = "linear-layer-1"
            };
            LinearLayer linearLayer2 = new LinearLayer(optimizerFactory, 1000, 500)
            {
                Name = "linear-layer-2"
            };
            LinearLayer linearLayer3 = new LinearLayer(optimizerFactory, 500, 1)
            {
                Name = "linear-layer-3"
            };
            SigmoidLayer sigmoid1 = new SigmoidLayer();
            SigmoidLayer sigmoid2 = new SigmoidLayer();
            SigmoidLayer sigmoid3 = new SigmoidLayer();
            ReLULayer reLU1 = new ReLULayer();
            ReLULayer reLU2 = new ReLULayer();
            ReLULayer reLU3 = new ReLULayer();
            this.Layers.Add(linearLayer1);
            // this.Layers.Add(sigmoid1);
            this.Layers.Add(reLU1);
            this.Layers.Add(linearLayer2);
            // this.Layers.Add(sigmoid2);
            // this.Layers.Add(reLU2);
            this.Layers.Add(linearLayer3);
            // this.Layers.Add(sigmoid3);
            // this.Layers.Add(reLU3);
        }
    }
}
