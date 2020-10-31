using NumSharp;

namespace NumSharpNetwork.Shared.Optimizers
{
    public class BasicOptimizer : IOptimizer
    {
        public NDArray LearningRate { get; set; }
        public NDArray Regularization { get; set; } = 0.1;
        public BasicOptimizer()
        {
        }
        public NDArray Optimize(NDArray weights, NDArray lossWeightGradients, bool isRegularization)
        {
            if (isRegularization)
            {
                return weights - this.LearningRate * (lossWeightGradients + weights * this.Regularization);
            }
            return weights - this.LearningRate * lossWeightGradients;
        }
    }
}
