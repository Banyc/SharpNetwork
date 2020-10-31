using NumSharp;

namespace NumSharpNetwork.Shared.Networks
{
    public interface ILayer
    {
        string Name { get; set; }
        bool IsTrainMode { get; set; }

        // feedforward
        NDArray FeedForward(NDArray input);
        // backpropagation
        NDArray BackPropagate(NDArray resultLossGradient);
        // save necessary state on disk
        void Save(string folderPath);
        // load state on disk if file exists
        void Load(string folderPath);
    }
}
