using Numpy;

namespace NumSharpNetwork.Shared.LossFunctions
{
    public enum CrossEntropyLabelType
    {
        OneHot,
        Embedded,
    }

    public class CrossEntropy : ILossFunction
    {
        public CrossEntropyLabelType LabelType { get; set; } = CrossEntropyLabelType.OneHot;

        // result.shape = [batchSize, numPropabilities]
        public NDarray GetLoss(NDarray result, NDarray label)
        {
            switch (this.LabelType)
            {
                case CrossEntropyLabelType.OneHot:
                    return GetLossOneHot(result, label);
                case CrossEntropyLabelType.Embedded:
                    return GetLossOneHot(result, GetOneHotLabel(result, label));
                default:
                    return null;
            }
        }

        // embeddedLabel.shape = [batchSize, 1]
        private NDarray GetOneHotLabel(NDarray result, NDarray embeddedLabel)
        {
            int batchSize = result.shape.Dimensions[0];
            int numClasses = result.shape.Dimensions[1];

            NDarray oneHotLabel = np.zeros_like(result);

            int batchIndex;
            // for each batch
            for (batchIndex = 0; batchIndex < batchSize; batchIndex++)
            {
                NDarray embedded = embeddedLabel[$"{batchIndex}"];
                int i;
                // for each embedded classIndex in the same batch
                for (i = 0; i < embedded.size; i++)
                {
                    int classIndex = embedded[$"{i}"].asscalar<int>();
                    oneHotLabel[$"{batchIndex}, {classIndex}"] = np.asarray(1);
                }
            }

            return oneHotLabel;
        }

        // label.shape = [batchSize, numClasses]
        private NDarray GetLossOneHot(NDarray result, NDarray label)
        {
            // In practice though, just in case our network returns a value of xi to close to zero, we clip its value to a minimum of 1e−8 (usually).
            NDarray clippedResult = result.clip(np.asarray(0.00000001), null);

            NDarray classlosses = label * -np.log(clippedResult);
            NDarray loss = classlosses.sum(1);
            return loss;
        }

        public NDarray GetLossResultGradient(NDarray result, NDarray label)
        {
            switch (this.LabelType)
            {
                case CrossEntropyLabelType.OneHot:
                    return GetLossResultGradientOneHot(result, label);
                case CrossEntropyLabelType.Embedded:
                    return GetLossResultGradientOneHot(result, GetOneHotLabel(result, label));
                default:
                    return null;
            }
        }

        // https://sgugger.github.io/a-simple-neural-net-in-numpy.html
        private NDarray GetLossResultGradientOneHot(NDarray result, NDarray label)
        {
            // In practice though, just in case our network returns a value of xi to close to zero, we clip its value to a minimum of 1e−8 (usually).
            // if result_i close to zero, the lossResultGradient_i will be -inf
            // -inf will cause the spread of NaN
            // NaN is infectious
            NDarray clippedResult = result.clip(np.asarray(0.00000001), null);

            // lossResultGradient = - result_i * (1 / label_i)
            NDarray lossResultGradient = np.where(label >= 1, -1 / clippedResult, np.asarray(0));
            return lossResultGradient;
        }
    }
}
