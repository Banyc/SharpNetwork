using System;
using System.Collections.Generic;
using System.IO;
using Numpy;
using NumSharpNetwork.Shared.Optimizers;

namespace NumSharpNetwork.Shared.Networks
{
    // the latest records of parameters and results of feedforward process
    public class LinearLayerRecord
    {
        // Input.shape := [batchSize, inputSize]
        public NDarray Input { get; set; }
        // Weight.shape := [batchSize, outputSize]
        public NDarray Weights { get; set; }
        // ForwardResult.shape := [batchSize, outputSize]
        public NDarray ForwardResult { get; set; }
    }

    public class LinearLayer : ILayer
    {
        public string Name { get; set; }
        public bool IsTrainMode { get; set; } = true;
        // Weights.shape := [inputSize, outputSize]
        public NDarray Weights { get; set; }
        // Biases.shape := [outputSize]
        public NDarray Biases { get; set; }
        public IOptimizer Optimizer { get; set; }

        // number of nodes connecting to a node in this layer
        public int InputSize { get; set; }
        // number of nodes in this layer
        public int OutputSize { get; set; }

        // the latest records of parameters and results of feedforward process
        private LinearLayerRecord Record { get; } = new LinearLayerRecord();

        public LinearLayer(IOptimizer optimizer, int inputSize, int outputSize, double weightScale = 0.001d)
        {
            this.Optimizer = optimizer;

            this.InputSize = inputSize;
            this.OutputSize = outputSize;

            // initialize biases and weights
            this.Biases = np.zeros(outputSize);
            this.Weights = np.random.randn(inputSize, outputSize) * weightScale;
        }

        // lossResultGradient := d_loss / d_result
        // result := output from the previous feedforward
        // lossResultGradient.shape := [batchSize, outputSize]
        // return d_loss / d_input
        public NDarray BackPropagate(NDarray lossResultGradient)
        {
            // lossBiasesGradient.shape := [outputSize]
            NDarray lossBiasesGradient = np.sum(lossResultGradient, 0);
            // lossWeightsGradient.shape := [inputSize, outputSize]
            NDarray lossWeightsGradient = np.matmul(this.Record.Input.T, lossResultGradient);
            // lossInputGradient.shape := [batchSize, inputSize]
            NDarray lossInputGradient = np.matmul(lossResultGradient, this.Record.Weights.T);

            this.Biases = this.Optimizer.Optimize(this.Biases, lossBiasesGradient, isAddRegularization: false);
            // this.Weights = this.Optimizer.Optimize(this.Weights, lossWeightsGradient, isAddRegularization: true);
            this.Weights = this.Optimizer.Optimize(this.Weights, lossWeightsGradient, isAddRegularization: false);

            return lossInputGradient;
        }

        // input := the output from the previous nodes
        // input.size := [batchSize, outputSize]
        public NDarray FeedForward(NDarray input)
        {
            NDarray result = np.matmul(input, this.Weights) + this.Biases;

            this.Record.Input = input;
            this.Record.ForwardResult = result;
            this.Record.Weights = this.Weights;

            return result;
        }

        public void Save(string folderPath)
        {
            Directory.CreateDirectory(folderPath);
            string statePath = Path.Combine(folderPath, $"{this.Name}.npy");
            np.save($"{statePath}.weights.npy", this.Weights);
            np.save($"{statePath}.biases.npy", this.Biases);
            // np.savez(statePath, new NDarray[]{this.Weights, this.Biases});
        }

        public void Load(string folderPath)
        {
            Directory.CreateDirectory(folderPath);
            string statePath = Path.Combine(folderPath, $"{this.Name}.npy");
            string weightPath = $"{statePath}.weights.npy";
            string biasesPath = $"{statePath}.biases.npy";
            if (File.Exists(weightPath))
            {
                this.Weights = np.load(weightPath);
            }
            if (File.Exists(biasesPath))
            {
                this.Biases = np.load(biasesPath);
            }
        }
    }
}
