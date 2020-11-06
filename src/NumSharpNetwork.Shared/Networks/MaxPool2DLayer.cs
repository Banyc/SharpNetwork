using System;
using Numpy;

namespace NumSharpNetwork.Shared.Networks
{
    public class MaxPool2DLayer : ILayer
    {
        public string Name { get; set; }
        public bool IsTrainMode { get; set; }
        public int KernelSize { get; set; }
        public int Stride { get; set; }
        private NDarray PreviousInput { get; set; }
        private NDarray PreviousResult { get; set; }

        public MaxPool2DLayer(int kernelSize = 3, int stride = 1)
        {
            this.KernelSize = kernelSize;
            this.Stride = stride;
        }

        public NDarray BackPropagate(NDarray lossResultGradient)
        {
            int poolHeight = this.KernelSize;
            int poolWidth = this.KernelSize;
            int batchSize = this.PreviousInput.shape.Dimensions[0];
            int inputChannels = this.PreviousInput.shape.Dimensions[1];
            int inputHeight = this.PreviousInput.shape.Dimensions[2];
            int inputWidth = this.PreviousInput.shape.Dimensions[3];

            int outputHeight = 1 + (inputHeight - poolHeight) / this.Stride;
            int outputWidth = 1 + (inputWidth - poolWidth) / this.Stride;

            NDarray lossInputGradient = np.zeros_like(this.PreviousInput);
            int heightIndex;
            int widthIndex;
            // iterating the result tensor of the feedforward process
            for (heightIndex = 0; heightIndex < outputHeight; heightIndex++)
            {
                for (widthIndex = 0; widthIndex < outputWidth; widthIndex++)
                {
                    NDarray receptiveField =
                        this.PreviousInput[$@"
                            :, 
                            :, 
                            {heightIndex * this.Stride} : {heightIndex * this.Stride + poolHeight}, 
                            {widthIndex * this.Stride} : {widthIndex * this.Stride + poolWidth}
                        "];

                    NDarray maskOnReceptiveField =
                        np.where(receptiveField >=
                            this.PreviousResult[$":, :, {heightIndex}, {widthIndex}"].reshape(batchSize, inputChannels, 1, 1),
                            np.asarray(1),
                            np.asarray(0));

                    NDarray lossInputGradientInReceptiveField =
                        maskOnReceptiveField * lossResultGradient[$":, :, {heightIndex}, {widthIndex}"].reshape(batchSize, inputChannels, 1, 1);

                    lossInputGradient[$@"
                            :, 
                            :, 
                            {heightIndex * this.Stride} : {heightIndex * this.Stride + poolHeight}, 
                            {widthIndex * this.Stride} : {widthIndex * this.Stride + poolWidth}
                        "] += lossInputGradientInReceptiveField;
                }
            }

            return lossInputGradient;
        }

        // https://leimao.github.io/blog/Max-Pooling-Backpropagation/
        // input.shape = [batchSize, inputChannels, inputHeight, inputWidth]
        public NDarray FeedForward(NDarray input)
        {
            int poolHeight = this.KernelSize;
            int poolWidth = this.KernelSize;
            int batchSize = input.shape.Dimensions[0];
            int inputChannels = input.shape.Dimensions[1];
            int inputHeight = input.shape.Dimensions[2];
            int inputWidth = input.shape.Dimensions[3];

            int outputHeight = 1 + (inputHeight - poolHeight) / this.Stride;
            int outputWidth = 1 + (inputWidth - poolWidth) / this.Stride;

            NDarray result = np.empty(batchSize, inputChannels, outputHeight, outputWidth);
            int heightIndex;
            int widthIndex;
            for (heightIndex = 0; heightIndex < outputHeight; heightIndex++)
            {
                for (widthIndex = 0; widthIndex < outputWidth; widthIndex++)
                {
                    NDarray receptiveField =
                        input[$@"
                            :, 
                            :, 
                            {heightIndex * this.Stride} : {heightIndex * this.Stride + poolHeight}, 
                            {widthIndex * this.Stride} : {widthIndex * this.Stride + poolWidth}
                        "];
                    result[$":, :, {heightIndex}, {widthIndex}"] = receptiveField.max(new int[] { 2, 3 });
                }
            }

            this.PreviousInput = input;
            this.PreviousResult = result;

            return result;
        }

        public void Load(string folderPath)
        {
        }

        public void Save(string folderPath)
        {
        }
    }
}
