using Numpy;

namespace NumSharpNetwork.Shared.LossFunctions
{
    public interface ILossFunction
    {
        // result := the {prediction/output value} from the last layer
        NDarray GetLoss(NDarray result, NDarray label);
        NDarray GetLossResultGradient(NDarray result, NDarray label);
    }
}
