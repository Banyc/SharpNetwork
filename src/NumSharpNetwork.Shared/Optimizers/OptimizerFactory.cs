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
        public double? LearningRate { get; set; } = null;
        public double? Momentum { get; set; } = null;

        public IOptimizer GetOptimizer()
        {
            return GetOptimizer(this.Type);
        }

        public IOptimizer GetOptimizer(OptimizerType type)
        {
            return type switch
            {
                OptimizerType.StochasticGradientDescent => GetStochasticGradientDescentOptimizer(),
                OptimizerType.SGDMomentum => GetSGDMomentumOptimizer(),
                _ => null,
            };
        }

        public StochasticGradientDescent GetStochasticGradientDescentOptimizer()
        {
            StochasticGradientDescent optimizer = new StochasticGradientDescent();
            if (this.LearningRate != null)
            {
                optimizer.LearningRate = this.LearningRate.Value;
            }
            return optimizer;
        }

        public SGDMomentum GetSGDMomentumOptimizer()
        {
            SGDMomentum sGDMomentum = new SGDMomentum();
            if (this.LearningRate != null)
            {
                sGDMomentum.LearningRate = this.LearningRate.Value;
            }
            if (this.Momentum != null)
            {
                sGDMomentum.Momentum = this.Momentum.Value;
            }
            return sGDMomentum;
        }
    }
}
