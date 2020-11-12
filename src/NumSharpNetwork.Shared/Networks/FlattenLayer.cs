using Numpy;

namespace NumSharpNetwork.Shared.Networks
{
    public class FlattenLayer : ILayer
    {
        public string Name { get; set; } = "Flatten";
        public bool IsTrainMode { get; set; }
        public NDarray PreviousInput { get; set; }

        public NDarray BackPropagate(NDarray lossResultGradient)
        {
            return lossResultGradient.reshape(
                this.PreviousInput.shape[0],
                this.PreviousInput.shape[1],
                this.PreviousInput.shape[2],
                this.PreviousInput.shape[3]
            );
        }

        // input.shape = [batchSize, channels, height, width]
        // return shape = [batchSize, channels * height * width]
        public NDarray FeedForward(NDarray input)
        {
            int batchSize = input.shape.Dimensions[0];
            NDarray result = input.reshape(batchSize, -1);

            this.PreviousInput = input;

            return result;
        }

        public void Load(string folderPath)
        {
        }

        public void Save(string folderPath)
        {
        }

        public override string ToString()
        {
            return $"{this.Name}";
        }
    }
}
