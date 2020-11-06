using System.Collections.Generic;
using Numpy;

namespace NumSharpNetwork.Shared.Networks.Wrappers
{
    public abstract class NetworkWrapper : ILayer
    {
        protected List<ILayer> Layers { get; } = new List<ILayer>();
        public virtual string Name { get; set; }
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

        public NetworkWrapper()
        {
        }

        public NDarray FeedForward(NDarray input)
        {
            NDarray result = input;
            foreach (ILayer layer in this.Layers)
            {
                result = layer.FeedForward(result);
                // DEBUG ONLY
                // check
                if (np.isnan(result).any())
                {
                    throw new System.Exception("NaN detected!");
                }
            }
            return result;
        }

        public NDarray BackPropagate(NDarray lossResultGradient)
        {
            NDarray result = lossResultGradient;
            int i;
            for (i = this.Layers.Count - 1; i >= 0; i--)
            {
                ILayer layer = this.Layers[i];
                result = layer.BackPropagate(result);
                // DEBUG ONLY
                // check
                if (np.isnan(result).any())
                {
                    throw new System.Exception("NaN detected!");
                }
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
