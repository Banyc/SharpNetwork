using Numpy;

namespace NumSharpNetwork.Shared.Networks
{
    public interface ILayer
    {
        string Name { get; set; }
        bool IsTrainMode { get; set; }

        // feedforward
        NDarray FeedForward(NDarray input);
        // backpropagation
        NDarray BackPropagate(NDarray resultLossGradient);
        // save necessary state on disk
        void Save(string folderPath);
        // load state on disk if file exists
        void Load(string folderPath);
    }
}
