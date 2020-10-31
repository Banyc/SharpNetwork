using NumSharp;

namespace NumSharpNetwork.Shared.Networks
{
    public interface ILayer
    {
        string Name { get; set; }
        bool IsTrainMode { get; set; }

        NDArray Forward(NDArray input);
        NDArray Backward(NDArray resultLossGradient);
        void Save(string folderPath);
        void Load(string folderPath);
    }
}
