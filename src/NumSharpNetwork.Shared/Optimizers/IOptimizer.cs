using Numpy;

namespace NumSharpNetwork.Shared.Optimizers
{
    public interface IOptimizer
    {
        double LearningRate { get; set; }
        NDarray Optimize(NDarray weights, NDarray lossWeightGradients);
    }
}
