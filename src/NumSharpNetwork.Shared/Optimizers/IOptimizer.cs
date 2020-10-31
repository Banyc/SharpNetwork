using Numpy;

namespace NumSharpNetwork.Shared.Optimizers
{
    public interface IOptimizer
    {
        NDarray LearningRate { get; set; }
        NDarray Optimize(NDarray weights, NDarray weightGradients, bool isRegularization);
    }
}
