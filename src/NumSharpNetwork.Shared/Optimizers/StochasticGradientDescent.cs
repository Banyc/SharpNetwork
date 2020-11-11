using Numpy;

namespace NumSharpNetwork.Shared.Optimizers
{
    // SGD
    public class StochasticGradientDescent : IOptimizer
    {
        public double LearningRate { get; set; } = 0.01;
        public NDarray Regularization { get; set; } = np.asarray(0.1);
        public StochasticGradientDescent()
        {
        }
        public NDarray Optimize(NDarray weights, NDarray lossWeightGradients)
        {
            // // DEBUG ONLY
            // // check
            // if (np.isnan(weights).any())
            // {
            //     throw new System.Exception("NaN detected!");
            // }
            NDarray newWeights = null;
            newWeights = weights - this.LearningRate * lossWeightGradients;
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
