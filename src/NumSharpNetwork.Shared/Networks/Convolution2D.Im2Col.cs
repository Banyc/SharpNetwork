using System.Runtime.CompilerServices;
using Numpy;

namespace NumSharpNetwork.Shared.Networks
{
    public partial class Convolution2D
    {
        private NDarray PreviousColumns { get; set; }

        // https://zhuanlan.zhihu.com/p/63974249
        public NDarray Im2ColFeedForward(NDarray input)
        {
            NDarray paddedInput = GetPaddedInput(input);

            int batchSize = input.shape[0];
            int paddedInputHeight = paddedInput.shape[2];
            int paddedInputWidth = paddedInput.shape[3];

            int resultHeight = (paddedInputHeight - this.FilterHeight + 1) / this.Stride;
            int resultWidth = (paddedInputWidth - this.FilterWidth + 1) / this.Stride;

            // number of columns := batchSize * resultHeight * resultWidth
            // size of each column := inputChannels * filterHeight * filterWidth
            NDarray columns = GetColumns(paddedInput, resultHeight, resultWidth);

            NDarray alignedWeights = this.FilterWeights.reshape(this.OutputChannels, -1);

            // [batchSize * resultHeight * resultWidth, outputChannels]
            NDarray multiply = np.dot(columns, alignedWeights.T);
            multiply = multiply.reshape(batchSize, resultHeight, resultWidth, this.OutputChannels);
            // multiply = multiply.transpose(0, 3, 1, 2);
            multiply = np.transpose(multiply, new int[] { 0, 3, 1, 2 });

            NDarray result =
                multiply.reshape(batchSize, OutputChannels, resultHeight, resultWidth) +
                this.Biases.reshape(1, -1, 1, 1);

            this.Record.Input = input;
            this.Record.PaddedInput = paddedInput;
            this.Record.Weights = this.FilterWeights;
            this.Record.ForwardResult = result;
            this.PreviousColumns = columns;

            return result;
        }

        // lossResultGradient.shape = [N, outputChannels, H, W]
        // return shape = [N, inputChannels, H, W]
        public NDarray Im2ColBackPropagate(NDarray lossResultGradient)
        {
            int batchSize = this.Record.Input.shape[0];
            int resultHeight = lossResultGradient.shape[2];
            int resultWidth = lossResultGradient.shape[3];
            NDarray lossBiasesGradient = lossResultGradient.sum(new int[] { 0, 2, 3 });

            // d_loss / d_multiply
            NDarray alignedWeights = this.Record.Weights.reshape(this.OutputChannels, -1);
            // NDarray lossMultiplyGradient = lossResultGradient.transpose(0, 2, 3, 1);
            NDarray lossMultiplyGradient = np.transpose(lossResultGradient, new int[] { 0, 2, 3, 1 });
            lossMultiplyGradient = lossMultiplyGradient.reshape(batchSize * resultHeight * resultWidth, this.OutputChannels);

            // d_loss / d_columns
            NDarray lossColumnsGradient = np.matmul(lossMultiplyGradient, alignedWeights);
            NDarray lossPaddedInputGradient = GetPaddedInputFromColumns(lossColumnsGradient, resultHeight, resultWidth, this.Record.PaddedInput.shape.Dimensions);
            NDarray lossInputGradient = GetUnpaddedInput(lossPaddedInputGradient);

            // d_loss / d_w
            NDarray lossAlignedWeightsGradient = np.matmul(this.PreviousColumns.T, lossMultiplyGradient);
            NDarray lossWeightsGradient = lossAlignedWeightsGradient.reshape(this.OutputChannels, this.InputChannels, this.FilterHeight, this.FilterWidth);

            // update
            this.Biases = this.BiasesOptimizer.Optimize(this.Biases, lossBiasesGradient);
            this.FilterWeights = this.WeightsOptimizer.Optimize(this.FilterWeights, lossWeightsGradient);

            return lossInputGradient;
        }

        // number of columns := batchSize * resultHeight * resultWidth
        // size of each column := inputChannels * filterHeight * filterWidth
        private NDarray GetColumns(NDarray paddedInput, int resultHeight, int resultWidth)
        {
            int batchSize = paddedInput.shape[0];
            // number of columns := batchSize * resultHeight * resultWidth
            // size of each column := inputChannels * filterHeight * filterWidth
            NDarray columns = np.empty(batchSize * resultHeight * resultWidth, this.InputChannels * this.FilterHeight * this.FilterWidth);

            // get columns from input/images
            int heightIndex;
            int widthIndex;
            for (heightIndex = 0; heightIndex < resultHeight; heightIndex++)
            {
                int startHeight = heightIndex * this.Stride;
                int endHeight = startHeight + this.FilterHeight;
                for (widthIndex = 0; widthIndex < resultWidth; widthIndex++)
                {
                    int startWidth = widthIndex * this.Stride;
                    int endWidth = startWidth + this.FilterWidth;

                    NDarray receptiveField = paddedInput[$":, :, {startHeight}:{endHeight}, {startWidth}:{endWidth}"];
                    // column.shape = [batchSize, stripeOfReceptiveField]
                    // stripeOfReceptiveField := inputChannels * filterHeight * filterWidth
                    NDarray column = receptiveField.reshape(batchSize, -1);

                    columns[$@"
                        {heightIndex * resultWidth + widthIndex}::{resultHeight * resultWidth}, 
                        :
                    "] = column;
                }
            }
            return columns;
        }

        // total reverse of `GetColumns`
        private NDarray GetPaddedInputFromColumns(NDarray columns, int resultHeight, int resultWidth, int[] previousPaddedInputShape)
        {
            int batchSize = previousPaddedInputShape[0];
            NDarray paddedInput = np.zeros(previousPaddedInputShape);

            int heightIndex;
            int widthIndex;
            for (heightIndex = 0; heightIndex < resultHeight; heightIndex++)
            {
                int startHeight = heightIndex * this.Stride;
                int endHeight = startHeight + this.FilterHeight;
                for (widthIndex = 0; widthIndex < resultWidth; widthIndex++)
                {
                    int startWidth = widthIndex * this.Stride;
                    int endWidth = startWidth + this.FilterWidth;

                    // column.shape = [batchSize, stripeOfReceptiveField]
                    // stripeOfReceptiveField := inputChannels * filterHeight * filterWidth
                    NDarray column =
                        columns[$@"
                            {heightIndex * resultWidth + widthIndex}::{resultHeight * resultWidth}, 
                            :
                        "];

                    NDarray receptiveField = column.reshape(batchSize, this.InputChannels, this.FilterHeight, this.FilterWidth);

                    paddedInput[$":, :, {startHeight}:{endHeight}, {startWidth}:{endWidth}"] += receptiveField;
                }
            }

            return paddedInput;
        }
    }
}
