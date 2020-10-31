using NumSharp;

namespace NumSharpNetwork.Shared.Optimizers
{
    public interface IOptimizer
    {
        NDArray LearningRate { get; set; }
        NDArray Optimize(NDArray weights, NDArray weightGradients, bool isRegularization);
    }
}
