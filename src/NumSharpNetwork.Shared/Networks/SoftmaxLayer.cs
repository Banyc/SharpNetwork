using System;
using Numpy;

namespace NumSharpNetwork.Shared.Networks
{
    public class SoftmaxLayer : ILayer
    {
        public string Name { get; set; }
        public bool IsTrainMode { get; set; }
        private NDarray PreviousResult { get; set; }

        public NDarray BackPropagate(NDarray lossResultGradient)
        {
            // https://sgugger.github.io/a-simple-neural-net-in-numpy.html
            NDarray lossInputGradient = this.PreviousResult * (
                lossResultGradient - (
                    lossResultGradient * this.PreviousResult
                ).sum(1).reshape(-1, 1)
            );

            // // zllz4
            // int batchSize = lossResultGradient.shape.Dimensions[0];
            // int numScores = lossResultGradient.shape.Dimensions[1];
            // NDarray diagonal = np.empty(batchSize, numScores, numScores);
            // for (int i = 0; i < batchSize; i++)
            // {
            //     diagonal[i] = np.diag(this.PreviousResult[i]);
            // }

            // NDarray lossInputGradient = np.sum(
            //     lossResultGradient.reshape(batchSize, -1, 1) *
            //     (
            //         diagonal -
            //         np.matmul(this.PreviousResult.reshape((batchSize, -1, 1)),
            //             this.PreviousResult.reshape((batchSize, 1, -1))
            //         )
            //     ), 1
            // );
            // Console.WriteLine(lossInputGradient);
            // Console.WriteLine(lossInputGradient);
            return lossInputGradient;
        }

        // input := logits
        // input.shape = [batchSize, numScores]
        // return shape = [batchSize, numScores]
        public NDarray FeedForward(NDarray input)
        {
            // // DEBUG ONLY
            // // check
            // if (np.isnan(input).any())
            // {
            //     throw new System.Exception("NaN detected!");
            // }

            // np.exp is not stable because it has Inf. So you should subtract maximum in x.
            // NDarray exps = np.exp(input - input.max());  // once input.max() == Inf, the whole exps will be 0's
            NDarray exps = np.exp(input - input.max(new int[] { 1 }).reshape(-1, 1));

            this.PreviousResult = exps / np.reshape(np.sum(exps, 1), new int[] { -1, 1 });

            // // DEBUG ONLY
            // // check
            // if (np.isnan(this.PreviousResult).any())
            // {
            //     // input might be too big.
            //     throw new System.Exception("NaN detected!");
            // }

            return this.PreviousResult;
        }

        public void Load(string folderPath)
        {
        }

        public void Save(string folderPath)
        {
        }
    }
}
