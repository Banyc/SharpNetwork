using Numpy;

namespace NumSharpNetwork.Shared.Optimizers
{
    public class OptimizerFactory
    {
        public StochasticGradientDescent GetStochasticGradientDescentOptimizer()
        {
            StochasticGradientDescent optimizer = new StochasticGradientDescent()
            {
                // LearningRate = 0.0001
                LearningRate = 0.001
                // LearningRate = 0.01
            };
            return optimizer;
        }

        public SGDMomentum GetSGDMomentumOptimizer()
        {
            SGDMomentum sGDMomentum = new SGDMomentum();
            return sGDMomentum;
        }
    }
}
