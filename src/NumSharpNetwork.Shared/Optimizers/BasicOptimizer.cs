using Numpy;

namespace NumSharpNetwork.Shared.Optimizers
{
    public class BasicOptimizer : IOptimizer
    {
        public NDarray LearningRate { get; set; }
        public NDarray Regularization { get; set; } = np.asarray(0.1);
        public BasicOptimizer()
        {
        }
        public NDarray Optimize(NDarray weights, NDarray lossWeightGradients, bool isAddRegularization)
        {
            if (isAddRegularization)
            {
                return weights - this.LearningRate * (lossWeightGradients + weights * this.Regularization);
            }
            return weights - this.LearningRate * lossWeightGradients;
        }
    }
}
