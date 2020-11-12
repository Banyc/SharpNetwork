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
        public string Name { get; set; } = "LinearLayer";
        public bool IsTrainMode { get; set; } = true;
        // Weights.shape := [inputSize, outputSize]
        public NDarray Weights { get; set; }
        // Biases.shape := [outputSize]
        public NDarray Biases { get; set; }
        public IOptimizer WeightsOptimizer { get; set; }
        public IOptimizer BiasesOptimizer { get; set; }

        // number of nodes connecting to a node in this layer
        public int InputSize { get; set; }
        // number of nodes in this layer
        public int OutputSize { get; set; }

        // the latest records of parameters and results of feedforward process
        private LinearLayerRecord Record { get; } = new LinearLayerRecord();

        public LinearLayer(OptimizerFactory optimizerFactory, int inputSize, int outputSize, double weightScale = 0.001d, string name = "linearLayer")
        {
            this.WeightsOptimizer = optimizerFactory.GetOptimizer();
            this.BiasesOptimizer = optimizerFactory.GetOptimizer();

            this.InputSize = inputSize;
            this.OutputSize = outputSize;
            this.Name = name;

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
            // two algorithms below. Either way could work.

            // zllz4
            int batchSize = this.Record.Input.shape[0];
            // lossBiasesGradient.shape := [outputSize]
            NDarray lossBiasesGradient = np.sum(lossResultGradient, 0);
            // NDarray lossBiasesGradient = np.mean(lossResultGradient, 0);  // the mean process has been implemented at CrossEntropy
            // lossWeightsGradient.shape := [inputSize, outputSize]
            NDarray lossWeightsGradient = np.matmul(this.Record.Input.T, lossResultGradient);
            // NDarray lossWeightsGradient = np.matmul(this.Record.Input.T, lossResultGradient) / batchSize;  // the mean process has been implemented at CrossEntropy
            // lossInputGradient.shape := [batchSize, inputSize]
            NDarray lossInputGradient = np.matmul(lossResultGradient, this.Record.Weights.T);

            // // https://sgugger.github.io/a-simple-neural-net-in-numpy.html
            // int batchSize = this.Record.Input.shape[0];
            // // lossBiasesGradient.shape := [outputSize]
            // // NDarray lossBiasesGradient = lossResultGradient.mean(0);  // the mean process has been implemented at CrossEntropy
            // NDarray lossBiasesGradient = lossResultGradient.sum(0);
            // // lossWeightsGradient.shape := [inputSize, outputSize]
            // NDarray lossWeightsGradient = np.matmul(
            //     this.Record.Input.reshape(batchSize, -1, 1),
            //     lossResultGradient.reshape(batchSize, 1, -1)
            // // ).mean(0);
            // ).sum(0);  // the mean process has been implemented at CrossEntropy
            // // lossInputGradient.shape := [batchSize, inputSize]
            // NDarray lossInputGradient = np.matmul(lossResultGradient, this.Record.Weights.T);

            this.Biases = this.BiasesOptimizer.Optimize(this.Biases, lossBiasesGradient);
            this.Weights = this.WeightsOptimizer.Optimize(this.Weights, lossWeightsGradient);

            return lossInputGradient;
        }

        // input := the output from the previous nodes
        // input.size := [batchSize, outputSize]
        public NDarray FeedForward(NDarray input)
        {
            NDarray result = np.dot(input, this.Weights) + this.Biases;

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

        public override string ToString()
        {
            return $"{this.Name} [{this.InputSize}, {this.OutputSize}]";
        }
    }
}
