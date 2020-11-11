using Numpy;

namespace NumSharpNetwork.Shared.Optimizers
{
    // Stochastic Gradient Descent with momentum
    public class SGDMomentum : IOptimizer
    {
        public double LearningRate { get; set; } = 0.01;
        public double Momentum { get; set; } = 0.9;
        private NDarray Velocity { get; set; } = null;

        public NDarray Optimize(NDarray weights, NDarray lossWeightGradients)
        {
            if (this.Velocity == null)
            {
                this.Velocity = np.zeros_like(weights);
            }

            this.Velocity = this.Velocity * this.Momentum - this.LearningRate * lossWeightGradients;
            NDarray newWeights = weights + this.Velocity;
            return newWeights;
        }
    }
}
