using System.Runtime.CompilerServices;
using System.Collections.Generic;
using NumSharp;
using NumSharpNetwork.Shared.Optimizers;

namespace NumSharpNetwork.Shared.Networks
{
    public class ThreeLinearLayers : ILayer
    {
        private List<LinearLayer> Layers { get; } = new List<LinearLayer>();
        public string Name { get; set; } = "three-layers";
        public bool IsTrainMode
        {
            get
            {
                return this.Layers[0].IsTrainMode;
            }
            set
            {
                foreach (var layer in this.Layers)
                {
                    layer.IsTrainMode = value;
                }
            }
        }

        public ThreeLinearLayers()
        {
            BasicOptimizer optimizer1 = new BasicOptimizer()
            {
                LearningRate = 0.1
            };
            LinearLayer layer1 = new LinearLayer(optimizer1, 10, 1000)
            {
                Name = "linear-layer-1"
            };
            BasicOptimizer optimizer2 = new BasicOptimizer()
            {
                LearningRate = 0.1
            };
            LinearLayer layer2 = new LinearLayer(optimizer2, 1000, 500)
            {
                Name = "linear-layer-2"
            };
            BasicOptimizer optimizer3 = new BasicOptimizer()
            {
                LearningRate = 0.1
            };
            LinearLayer layer3 = new LinearLayer(optimizer3, 500, 1)
            {
                Name = "linear-layer-3"
            };
            this.Layers.Add(layer1);
            this.Layers.Add(layer2);
            this.Layers.Add(layer3);
        }

        public NDArray Forward(NDArray input)
        {
            NDArray result = input;
            foreach (ILayer layer in this.Layers)
            {
                result = layer.Forward(result);
            }
            return result;
        }

        public NDArray Backward(NDArray resultLossGradient)
        {
            NDArray result = resultLossGradient;
            int i;
            for (i = this.Layers.Count - 1; i >= 0; i--)
            {
                ILayer layer = this.Layers[i];
                result = layer.Backward(result);
            }
            return result;
        }

        public void Save(string folderPath)
        {
            foreach (var layer in this.Layers)
            {
                layer.Save(folderPath);
            }
        }

        public void Load(string folderPath)
        {
            foreach (var layer in this.Layers)
            {
                layer.Load(folderPath);
            }
        }
    }
}
