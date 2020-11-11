using Numpy;

namespace NumSharpNetwork.Shared.Optimizers
{
    public abstract class OptimizerBase : IOptimizer
    {
        public abstract double LearningRate { get; set; }
        public double WeightDecay { get; set; } = 0f;

        protected abstract NDarray Optimize(NDarray weights, NDarray lossWeightGradients);

        public NDarray Optimize(NDarray weights, NDarray lossWeightGradients, bool IsWeightDecay = false)
        {
            if (IsWeightDecay)
            {
                weights = GetDecayedWeights(weights);
            }
            return Optimize(weights, lossWeightGradients);
        }

        private NDarray GetDecayedWeights(NDarray weights)
        {
            return (1f - this.WeightDecay) * weights;
        }
    }
}
