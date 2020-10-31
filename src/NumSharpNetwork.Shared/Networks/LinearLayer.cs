using System;
using System.Collections.Generic;
using System.IO;
using NumSharp;
using NumSharpNetwork.Shared.Optimizers;

namespace NumSharpNetwork.Shared.Networks
{
    // the latest records of parameters and results of feedforward process
    public class LinearLayerRecord
    {
        // Input.shape := [batchSize, inputSize]
        public NDArray Input { get; set; }
        // Weight.shape := [batchSize, outputSize]
        public NDArray Weights { get; set; }
        // ForwardResult.shape := [batchSize, outputSize]
        public NDArray ForwardResult { get; set; }
    }

    public class LinearLayer : ILayer
    {
        public string Name { get; set; }
        public bool IsTrainMode { get; set; } = true;
        // Weights.shape := [inputSize, outputSize]
        public NDArray Weights { get; set; }
        // Biases.shape := [outputSize]
        public NDArray Biases { get; set; }
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
        public NDArray BackPropagate(NDArray lossResultGradient)
        {
            // lossBiasesGradient.shape := [outputSize]
            NDArray lossBiasesGradient = np.sum(lossResultGradient, 0);
            // lossWeightsGradient.shape := [inputSize, outputSize]
            NDArray lossWeightsGradient = np.matmul(this.Record.Input.T, lossResultGradient);
            // lossInputGradient.shape := [batchSize, inputSize]
            NDArray lossInputGradient = np.matmul(lossResultGradient, this.Record.Weights.T);

            this.Biases = this.Optimizer.Optimize(this.Biases, lossBiasesGradient, isRegularization: false);
            this.Weights = this.Optimizer.Optimize(this.Weights, lossWeightsGradient, isRegularization: true);

            return lossInputGradient;
        }

        // input := the output from the previous nodes
        // input.size := [batchSize, outputSize]
        public NDArray FeedForward(NDArray input)
        {
            NDArray result = np.matmul(input, this.Weights) + this.Biases;

            this.Record.Input = input;
            this.Record.ForwardResult = result;
            this.Record.Weights = this.Weights;

            return result;
        }

        public void Save(string folderPath)
        {
            Directory.CreateDirectory(folderPath);
            string statePath = Path.Combine(folderPath, $"{this.Name}.npy");
            Dictionary<string, Array> state = new Dictionary<string, Array>
            {
                ["weights"] = (Array)this.Weights,
                ["biases"] = (Array)this.Biases
            };
            np.Save_Npz(state, statePath);
        }

        public void Load(string folderPath)
        {
            Directory.CreateDirectory(folderPath);
            string statePath = Path.Combine(folderPath, $"{this.Name}.npy");
            if (!File.Exists(statePath))
            {
                return;
            }
            NpzDictionary<Array> loadedState = np.Load_Npz<Array>(statePath);
            this.Weights = loadedState["weights"];
            this.Biases = loadedState["biases"];
        }
    }
}
