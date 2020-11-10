using Numpy;

namespace NumSharpNetwork.Shared.Optimizers
{
    // SGD
    public class StochasticGradientDescent : IOptimizer
    {
        public NDarray LearningRate { get; set; }
        public NDarray Regularization { get; set; } = np.asarray(0.1);
        public BatchGradientDescent()
        {
        }
        public NDarray Optimize(NDarray weights, NDarray lossWeightGradients, bool isAddRegularization)
        {
            // // DEBUG ONLY
            // // check
            // if (np.isnan(weights).any())
            // {
            //     throw new System.Exception("NaN detected!");
            // }
            NDarray newWeights = null;
            if (isAddRegularization)
            {
                newWeights = weights - this.LearningRate * (lossWeightGradients + weights * this.Regularization);
            }
            else
            {
                newWeights = weights - this.LearningRate * lossWeightGradients;
            }
            // // DEBUG ONLY
            // // check
            // if (np.isnan(newWeights).any())
            // {
            //     throw new System.Exception("NaN detected!");
            // }
            return newWeights;
        }
    }
}
