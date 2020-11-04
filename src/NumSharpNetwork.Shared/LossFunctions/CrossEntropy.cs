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
            int classIndex;
            for (batchIndex = 0; batchIndex < batchSize; batchIndex++)
            {
                NDarray embedded = embeddedLabel[$"{batchIndex}"];
                for (classIndex = 0; classIndex < embedded.size; classIndex++)
                {
                    oneHotLabel[$"{batchIndex}, {classIndex}"] = np.asarray(1);
                }
            }

            return oneHotLabel;
        }

        // label.shape = [batchSize, numClasses]
        // https://sgugger.github.io/a-simple-neural-net-in-numpy.html
        private NDarray GetLossOneHot(NDarray result, NDarray label)
        {
            // In practice though, just in case our network returns a value of xi to close to zero, we clip its value to a minimum of 1e−8 (usually).
            NDarray clippedResult = result.clip(np.asarray(0.00000001), null);

            NDarray loss = np.where(np.asarray(label == np.asarray(1)), -np.log(clippedResult), -np.log(1 - clippedResult)).sum(1);
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
            NDarray lossResultGradient = np.where(np.asarray(label == np.asarray(1)), -1 / (result * np.log(np.asarray(2))), np.asarray(0));
            return lossResultGradient;
        }
    }
}
