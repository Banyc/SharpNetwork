using Numpy;

namespace NumSharpNetwork.Shared.Networks
{
    public class SigmoidLayer : ILayer
    {
        public string Name { get; set; } = "Sigmoid";
        public bool IsTrainMode { get; set; }
        public NDarray PreviousInput { get; set; }
        public NDarray PreviousResult { get; set; }

        public NDarray BackPropagate(NDarray lossResultGradient)
        {
            // S(x)
            NDarray sigmoidOnPreviousInput = this.PreviousResult;
            // d_S(x) / d_x = S(x) * (1 - S(x))
            NDarray resultInputGradient = sigmoidOnPreviousInput * (1 - sigmoidOnPreviousInput);
            // d_loss / d_x = (d_loss / d_S(x)) * (d_S(x) / d_x)
            NDarray lossInputGradient = lossResultGradient * resultInputGradient;
            return lossInputGradient;
        }

        public NDarray FeedForward(NDarray input)
        {
            this.PreviousResult = Sigmoid(input);
            return this.PreviousResult;
        }

        public void Load(string folderPath)
        {
        }

        public void Save(string folderPath)
        {
        }

        private NDarray Sigmoid(NDarray input)
        {
            return 1.0 / (1 + np.exp(-input));
        }
        
        public override string ToString()
        {
            return $"{this.Name}";
        }
    }
}
