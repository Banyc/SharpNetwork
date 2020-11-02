using Numpy;

namespace NumSharpNetwork.Shared.LossFunctions
{
    // MSE
    public class MeanSquaredError : ILossFunction
    {
        // result.shape = [batchSize, 1]
        public NDarray GetLoss(NDarray result, NDarray label)
        {
            // loss = 0.5 * sum( (solution - predict_i) ^ 2 ) / N
            NDarray loss = np.asarray(0.5) * np.mean(np.power(label - np.squeeze(result), np.asarray(2)));
            return loss;
        }

        // result.shape = [batchSize, 1]
        public NDarray GetLossResultGradient(NDarray result, NDarray label)
        {
            int batchSize = result.shape.Dimensions[0];
            // d_loss/d_predict_i = - (1/N) * (solution - predict_i)
            NDarray lossResultGradient = -np.reshape(label - np.squeeze(result), (batchSize, 1)) / batchSize;
            return lossResultGradient;
        }
    }
}
