using Numpy;

namespace NumSharpNetwork.Shared.Optimizers
{
    public class OptimizerFactory
    {
        public StochasticGradientDescent GetStochasticGradientDescentOptimizer()
        {
            StochasticGradientDescent optimizer = new StochasticGradientDescent()
            {
                // LearningRate = np.asarray(0.0001)
                LearningRate = np.asarray(0.001)
                // LearningRate = np.asarray(0.01)
            };
            return optimizer;
        }
    }
}
