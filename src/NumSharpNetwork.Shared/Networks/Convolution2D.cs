using Numpy;
using NumSharpNetwork.Shared.Optimizers;
using System.IO;

// https://cs231n.github.io/convolutional-networks/

namespace NumSharpNetwork.Shared.Networks
{
    public enum Convolution2DMode
    {
        Simple,
        Fast,
    }

    // the latest records of parameters and results of feedforward process
    public class Convolution2DRecord
    {
        public NDarray Input { get; set; }
        public NDarray PaddedInput { get; set; }
        public NDarray PaddedInput5D { get; set; }
        public NDarray Weights { get; set; }
        public NDarray FilterWeights5D { get; set; }
        public NDarray ForwardResult { get; set; }
    }

    public partial class Convolution2D : ILayer
    {
        public string Name { get; set; } = "Convolution2D";
        public bool IsTrainMode { get; set; }
        public int KernelSize { get; set; }
        public int Stride { get; set; }
        public bool IsIm2Col { get; set; } = false;

        // number of zeros to be padded to the input
        public int TopLeftPaddingSize
        {
            get
            {
                return ((this.KernelSize - 1) * this.Stride) / 2;
            }
        }
        public int BottomRightPaddingSize
        {
            get
            {
                int carry = (this.KernelSize - 1) * this.Stride % 2;
                return ((this.KernelSize - 1) * this.Stride / 2) + carry;
            }
        }
        // FilterWeights.shape := [outputChannels, inputChannels, kernelSize, kernelSize]
        public NDarray FilterWeights { get; set; }
        private int OutputChannels { get => this.FilterWeights.shape[0]; }
        private int InputChannels { get => this.FilterWeights.shape[1]; }
        private int FilterHeight { get => this.FilterWeights.shape[2]; }
        private int FilterWidth { get => this.FilterWeights.shape[3]; }

        // Biases.shape = [outputChannels]
        public NDarray Biases { get; set; }
        public IOptimizer WeightsOptimizer { get; set; }
        public IOptimizer BiasesOptimizer { get; set; }
        private Convolution2DRecord Record { get; set; } = new Convolution2DRecord();

        public Convolution2D(int inputChannels, int outputChannels, OptimizerFactory optimizerFactory, int kernelSize = 3, int stride = 1, double weightScale = 0.001d, string name = "convolution2D")
        {
            this.KernelSize = kernelSize;
            this.Stride = stride;
            this.WeightsOptimizer = optimizerFactory.GetOptimizer();
            this.BiasesOptimizer = optimizerFactory.GetOptimizer();
            this.Name = name;

            this.FilterWeights = np.random.randn(outputChannels, inputChannels, kernelSize, kernelSize) * weightScale;
            this.Biases = np.zeros(outputChannels);
        }

        // input.shape = [N, C, H, W]
        // N := batchSize
        // C := color channels (inputChannels)
        // H := image height
        // W := image width
        // return.shape = [N, outputChannels, H, W]
        public NDarray FeedForward(NDarray input)
        {
            if (this.IsIm2Col)
            {
                return Im2ColFeedForward(input);
            }
            // pad zeros on the borders of the input
            NDarray paddedInput = GetPaddedInput(input);

            int batchSize = input.shape[0];
            int paddedInputHeight = paddedInput.shape[2];
            int paddedInputWidth = paddedInput.shape[3];

            int resultHeight = (paddedInputHeight - this.FilterHeight + 1) / this.Stride;
            int resultWidth = (paddedInputWidth - this.FilterWidth + 1) / this.Stride;

            // allocate memory for result
            NDarray result = np.empty(
                // batch size
                batchSize,
                // output Channels
                this.OutputChannels,
                // height = (paddedInput.height - FilterWeights.height + 1) / stride
                resultHeight,
                // width = (paddedInput.width - FilterWeights.width + 1) / stride
                resultWidth
            );
            // // DEBUG ONLY
            // result = np.full_like(result, np.nan);

            // reshape to align the last three axises, preparing for multiplication
            // the size 1 in some axis could be considered as a scalar when that axis multiplies the same axis with a different size
            // ex: [5] * [2, 3] == [10, 15]  // in the same axis but with different sizes
            NDarray filterWeights5D = this.FilterWeights.reshape(
                1,
                this.OutputChannels,
                this.InputChannels,
                this.FilterHeight,
                this.FilterWidth
            );
            NDarray paddedInput5D = paddedInput.reshape(
                batchSize,
                1,
                this.InputChannels,
                paddedInputHeight,
                paddedInputWidth
            );

            int heightIndex;
            int widthIndex;
            // scan the whole-ass image
            for (heightIndex = 0; heightIndex < resultHeight; heightIndex++)
            {
                for (widthIndex = 0; widthIndex < resultWidth; widthIndex++)
                {
                    // cut the receptive field from the paddedInput
                    NDarray receptiveField = paddedInput5D[
                        $@"
                        :, 
                        :, 
                        :, 
                        {this.Stride * heightIndex}:{(this.Stride * heightIndex) + this.FilterHeight}, 
                        {this.Stride * widthIndex}:{(this.Stride * widthIndex) + this.FilterWidth}"
                    ];

                    // multiply the field with filterWeights
                    NDarray multiplyMediacy = receptiveField * filterWeights5D;
                    // ... and sum to a bar
                    // outputBar.shape = [batchSize, outputChannels, 1, 1, 1]
                    NDarray outputBar = np.sum(multiplyMediacy, new int[] { 2, 3, 4 });
                    // insert it to the result
                    result[$":, :, {heightIndex}, {widthIndex}"] = outputBar;
                }
            }

            // align biases to the outputChannels in result
            NDarray alignedBiases = this.Biases.reshape(1, -1, 1, 1);
            // add up result with biases
            result += alignedBiases;

            // save for backpropagation
            this.Record.Input = input;
            this.Record.PaddedInput = paddedInput;
            this.Record.Weights = this.FilterWeights;
            this.Record.FilterWeights5D = filterWeights5D;
            this.Record.PaddedInput5D = paddedInput5D;
            this.Record.ForwardResult = result;

            // // DEBUG ONLY
            // // check
            // if (np.isnan(result).any())
            // {
            //     throw new System.Exception("NaN detected!");
            // }

            return result;
        }

        public NDarray BackPropagate(NDarray lossResultGradient)
        {
            if (this.IsIm2Col)
            {
                return Im2ColBackPropagate(lossResultGradient);
            }
            int batchSize = this.Record.Input.shape[0];
            int outputChannels = this.Record.Weights.shape[0];
            int inputChannels = this.Record.Weights.shape[1];
            int filterHeight = this.Record.Weights.shape[2];
            int filterWidth = this.Record.Weights.shape[3];
            int paddedInputHeight = this.Record.PaddedInput.shape[2];
            int paddedInputWidth = this.Record.PaddedInput.shape[3];

            int resultHeight = (paddedInputHeight - filterHeight + 1) / this.Stride;
            int resultWidth = (paddedInputWidth - filterWidth + 1) / this.Stride;

            // lossWeightsGradient.shape = [outputChannels, inputChannels, filterHeight, filterWidth]
            NDarray lossWeightsGradient = np.zeros_like(this.Record.Weights);
            // paddedLossInputGradient.shape = [batchSize, inputChannels, paddedInputHeight, paddedInputWidth]
            NDarray paddedLossInputGradient = np.zeros_like(this.Record.PaddedInput);

            int heightIndex;
            int widthIndex;
            // scan the whole-ass image
            for (heightIndex = 0; heightIndex < resultHeight; heightIndex++)
            {
                for (widthIndex = 0; widthIndex < resultWidth; widthIndex++)
                {
                    // cut the receptive field from the paddedInput5D
                    NDarray receptiveField = this.Record.PaddedInput5D[
                        $@"
                        :, 
                        :, 
                        :, 
                        {this.Stride * heightIndex}:{(this.Stride * heightIndex) + filterHeight}, 
                        {this.Stride * widthIndex}:{(this.Stride * widthIndex) + filterWidth}"
                    ];
                    // cut the outputBar from lossResultGradient
                    NDarray outputBarFromLossResultGradient = lossResultGradient[
                        $@"
                        :, 
                        :, 
                        {heightIndex},
                        {widthIndex}"
                    ];
                    // align with filterWeights5D
                    NDarray outputBarFromLossResultGradient5D = outputBarFromLossResultGradient.reshape(batchSize, outputChannels, 1, 1, 1);

                    // d_loss/d_x = d_loss/d_result * w'
                    NDarray receptiveFieldFromLossInputGradient = np.sum(this.Record.FilterWeights5D * outputBarFromLossResultGradient5D, 1);
                    // since elements in {d_loss/d_result} to some element in {d_loss/d_x} is a many-to-one relationship,
                    // the partial gradient {d_loss/d_x} should be added up
                    paddedLossInputGradient[
                        $@"
                        :, 
                        :, 
                        {this.Stride * heightIndex}:{(this.Stride * heightIndex) + filterHeight},
                        {this.Stride * widthIndex}:{(this.Stride * widthIndex) + filterWidth}"
                    ] += receptiveFieldFromLossInputGradient;

                    // d_loss/d_w = x * d_loss/d_result
                    NDarray partialLossWeightsGradient = np.sum(receptiveField * outputBarFromLossResultGradient5D, 0);
                    // since elements in {d_loss/d_result} to some element in {d_loss/d_weight} is a many-to-one relationship,
                    // the partial gradient {d_loss/d_weight} should be added up
                    lossWeightsGradient += partialLossWeightsGradient;
                }
            }

            NDarray lossInputGradient = GetUnpaddedInput(paddedLossInputGradient);

            // since, in the feedforward process, the same biases is been used to produce many batches and many channels in the result,
            // the partial gradients should also be added up
            NDarray lossBiasesGradient = np.sum(lossResultGradient, new int[] { 0, 2, 3 });

            // update
            this.FilterWeights = this.WeightsOptimizer.Optimize(this.FilterWeights, lossWeightsGradient);
            this.Biases = this.BiasesOptimizer.Optimize(this.Biases, lossBiasesGradient);

            return lossInputGradient;
        }

        public void Save(string folderPath)
        {
            Directory.CreateDirectory(folderPath);
            string statePath = Path.Combine(folderPath, $"{this.Name}.npy");
            np.save($"{statePath}.weights.npy", this.FilterWeights);
            np.save($"{statePath}.biases.npy", this.Biases);
        }

        public void Load(string folderPath)
        {
            Directory.CreateDirectory(folderPath);
            string statePath = Path.Combine(folderPath, $"{this.Name}.npy");
            string weightPath = $"{statePath}.weights.npy";
            string biasesPath = $"{statePath}.biases.npy";
            if (File.Exists(weightPath))
            {
                this.FilterWeights = np.load(weightPath);
            }
            if (File.Exists(biasesPath))
            {
                this.Biases = np.load(biasesPath);
            }
        }

        private NDarray GetPaddedInput(NDarray input)
        {
            // pad zeros on the borders of the input
            int[,] padWidth = new int[,]{
                {0, 0},
                {0, 0},
                {this.TopLeftPaddingSize, this.BottomRightPaddingSize},
                {this.TopLeftPaddingSize, this.BottomRightPaddingSize}
            };
            NDarray pad_width = np.array(padWidth);
            // height and width are padded with zeros
            NDarray paddedInput = np.pad(input, pad_width, "constant");
            return paddedInput;
        }

        private NDarray GetUnpaddedInput(NDarray paddedInput)
        {
            return paddedInput[
                $@"
                :, 
                :, 
                {this.TopLeftPaddingSize}:{-this.BottomRightPaddingSize}, 
                {this.TopLeftPaddingSize}:{-this.BottomRightPaddingSize}"
            ];
        }

        public override string ToString()
        {
            return $"{this.Name} [{this.InputChannels}, {this.OutputChannels}]";
        }
    }
}
