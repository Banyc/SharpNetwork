using System.Net.Mail;
using System.Reflection.Emit;
using Numpy;

namespace NumSharpNetwork.Shared.Networks
{
    public class SigmoidLayer : ILayer
    {
        public string Name { get; set; }
        public bool IsTrainMode { get; set; }
        public NDarray PreviousInput { get; set; }

        public NDarray BackPropagate(NDarray lossResultGradient)
        {
            NDarray sigmoidOnPreviousInput = Sigmoid(this.PreviousInput);
            NDarray lossInputGradient = lossResultGradient * (sigmoidOnPreviousInput * (1 - sigmoidOnPreviousInput));
            return lossInputGradient;
        }

        public NDarray FeedForward(NDarray input)
        {
            this.PreviousInput = input;
            return Sigmoid(input);
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
    }
}
