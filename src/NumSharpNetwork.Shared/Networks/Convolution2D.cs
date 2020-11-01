using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;
using System;
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
        public NDarray Weights { get; set; }
        public NDarray ForwardResult { get; set; }
    }

    public class Convolution2D : ILayer
    {
        public string Name { get; set; }
        public bool IsTrainMode { get; set; }
        public int KernelSize { get; set; }
        public int Stride { get; set; }
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
        public NDarray Biases { get; set; }
        public IOptimizer Optimizer { get; set; }
        private Convolution2DRecord Record { get; set; } = new Convolution2DRecord();

        public Convolution2D(int inputChannels, int outputChannels, IOptimizer optimizer, int kernelSize = 3, int stride = 1, double weightScale = 0.001d)
        {
            this.KernelSize = kernelSize;
            this.Stride = stride;
            this.Optimizer = optimizer;

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

            int batchSize = input.shape.Dimensions[0];
            int outputChannels = this.FilterWeights.shape.Dimensions[0];
            int inputChannels = this.FilterWeights.shape.Dimensions[1];
            int filterHeight = this.FilterWeights.shape.Dimensions[2];
            int filterWidth = this.FilterWeights.shape.Dimensions[3];
            int paddedInputHeight = paddedInput.shape.Dimensions[2];
            int paddedInputWidth = paddedInput.shape.Dimensions[3];

            // allocate memory for result
            NDarray result = np.zeros(
                // batch size
                batchSize,
                // output Channels
                outputChannels,
                // height = (paddedInput.height - FilterWeights.height + 1) / stride
                (paddedInputHeight - filterHeight + 1) / this.Stride,
                // width = (paddedInput.width - FilterWeights.width + 1) / stride
                (paddedInputWidth - filterWidth + 1) / this.Stride
            );

            // reshape
            NDarray filterWeights5D = this.FilterWeights.reshape(
                1,
                outputChannels,
                inputChannels,
                filterHeight,
                filterWidth
            );
            NDarray paddedInput5D = paddedInput.reshape(
                batchSize,
                1,
                inputChannels,
                paddedInputHeight,
                paddedInputWidth
            );

            int heightIndex;
            int widthIndex;
            for (heightIndex = 0; heightIndex + 1 + filterHeight <= paddedInputHeight; heightIndex++)
            {
                for (widthIndex = 0; widthIndex + 1 + filterWidth <= paddedInputWidth; widthIndex++)
                {
                    // cut the receptive field from the paddedInput
                    NDarray receptiveField = paddedInput5D[
                        $@"
                        :, 
                        :, 
                        :, 
                        {this.Stride * heightIndex}:{(this.Stride * heightIndex) + filterHeight}, 
                        {this.Stride * widthIndex}:{(this.Stride * widthIndex) + filterWidth}"
                    ];

                    // multiply the field with filterWeights
                    NDarray multiplyMediacy = receptiveField * filterWeights5D;
                    // ... and sum to a bar
                    NDarray outputBar = np.sum(multiplyMediacy, new int[] { 2, 3, 4 });
                    // put it to the result
                    result[$":, :, {heightIndex}, {widthIndex}"] = outputBar;
                }
            }

            // align biases to the outputChannels in result
            NDarray alignedBiases = this.Biases.reshape(1, -1, 1, 1);
            // add up result with biases
            result += alignedBiases;

            // save for backpropagation
            this.Record.Input = input;
            this.Record.Weights = this.FilterWeights;
            this.Record.ForwardResult = result;

            return result;
        }

        public NDarray BackPropagate(NDarray lossResultGradient)
        {
            throw new System.NotImplementedException();
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
    }
}
