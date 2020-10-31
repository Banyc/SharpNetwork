using System;
using System.Collections.Generic;
using System.IO;
using NumSharp;
using NumSharpNetwork.Shared.Optimizers;

namespace NumSharpNetwork.Shared.Networks
{
    public class LinearLayerRecord
    {
        public NDArray Input { get; set; }
        public NDArray Weights { get; set; }
        public NDArray ForwardResult { get; set; }
    }

    public class LinearLayer : ILayer
    {
        public string Name { get; set; }
        public bool IsTrainMode { get; set; } = true;
        public NDArray Weights { get; set; }
        public NDArray Biases { get; set; }
        public IOptimizer Optimizer { get; set; }
        public int InputSize { get; set; }
        public int OutputSize { get; set; }

        private LinearLayerRecord Record { get; } = new LinearLayerRecord();

        public LinearLayer(IOptimizer optimizer, int inputSize, int outputSize, double weightScale = 0.001d)
        {
            this.Optimizer = optimizer;

            this.InputSize = inputSize;
            this.OutputSize = outputSize;

            this.Biases = np.zeros(outputSize);
            this.Weights = np.random.randn(inputSize, outputSize) * weightScale;
        }

        public NDArray Backward(NDArray lossResultGradient)
        {
            NDArray lossBiasesGradient = np.sum(lossResultGradient, 0);
            NDArray lossWeightsGradient = np.dot(this.Record.Input.T, lossResultGradient);
            NDArray lossInputGradient = np.dot(lossResultGradient, this.Record.Weights.T);

            this.Biases = this.Optimizer.Optimize(this.Biases, lossBiasesGradient, isRegularization: false);
            this.Weights = this.Optimizer.Optimize(this.Weights, lossWeightsGradient, isRegularization: true);

            return lossInputGradient;
        }

        public NDArray Forward(NDArray input)
        {
            NDArray result = np.dot(input, this.Weights) + this.Biases;

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
