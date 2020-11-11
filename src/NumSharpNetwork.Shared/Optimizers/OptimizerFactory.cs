using Numpy;

namespace NumSharpNetwork.Shared.Optimizers
{
    public enum OptimizerType
    {
        StochasticGradientDescent,
        SGDMomentum,
    }

    public class OptimizerFactory
    {
        public OptimizerType Type { get; set; } = OptimizerType.StochasticGradientDescent;

        public IOptimizer GetOptimizer()
        {
            return GetOptimizer(this.Type);
        }

        public static IOptimizer GetOptimizer(OptimizerType type)
        {
            return type switch
            {
                OptimizerType.StochasticGradientDescent => GetStochasticGradientDescentOptimizer(),
                OptimizerType.SGDMomentum => GetSGDMomentumOptimizer(),
                _ => null,
            };
        }

        public static StochasticGradientDescent GetStochasticGradientDescentOptimizer()
        {
            StochasticGradientDescent optimizer = new StochasticGradientDescent()
            {
                // LearningRate = 0.0001
                LearningRate = 0.001
                // LearningRate = 0.01
            };
            return optimizer;
        }

        public static SGDMomentum GetSGDMomentumOptimizer()
        {
            SGDMomentum sGDMomentum = new SGDMomentum();
            return sGDMomentum;
        }
    }
}
