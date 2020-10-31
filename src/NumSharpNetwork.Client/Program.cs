using System;
using System.Collections.Generic;
using System.IO;
using Numpy;
using NumSharpNetwork.Shared.Networks;

namespace NumSharpNetwork.Client
{
    class Program
    {
        static void Main(string[] args)
        {
            // a hidden set of weights that is used to generate dataset
            NDarray weights = np.random.randn(10);
            const int batchSize = 200;

            ThreeLinearLayers layers = new ThreeLinearLayers();

            const string statePath = "trainings";
            Train(layers, batchSize, weights, statePath);
        }

        static Dictionary<string, NDarray> LoadState(string folderPath)
        {
            Directory.CreateDirectory(folderPath);
            string statePath = Path.Combine(folderPath, $"test1.npy");
            Dictionary<string, NDarray> stateDict = new Dictionary<string, NDarray>();

            if (File.Exists($"{statePath}.weights.npy"))
            {
                stateDict["weights"] = np.load($"{statePath}.weights.npy");
            }
            if (File.Exists($"{statePath}.step.npy"))
            {
                stateDict["step"] = np.load($"{statePath}.step.npy");
            }
            return stateDict;
        }

        static void SaveState(string folderPath, Dictionary<string, NDarray> state)
        {
            Directory.CreateDirectory(folderPath);
            string statePath = Path.Combine(folderPath, $"test1.npy");
            np.save($"{statePath}.weights.npy", state["weights"]);
            np.save($"{statePath}.step.npy", state["step"]);
            // np.savez(statePath, kwds: state);
            // np.savez(statePath, new NDarray[]{state["weights"], state["step"]});
        }

        static (NDarray data, NDarray label) GetDataset(int batchSize, NDarray weights)
        {
            NDarray data = np.random.randn(batchSize, 10);
            NDarray label = np.sum(weights * data, 1);
            return (data, label);
        }

        static void Train(ILayer layer, int batchSize, NDarray weights, string statePath)
        {
            // load state
            Dictionary<string, NDarray> trainState = LoadState(statePath);
            if (trainState.ContainsKey("weights"))
            {
                weights = trainState["weights"];
            }
            else
            {
                trainState["weights"] = weights;
            }
            layer.Load(statePath);

            // restore step number
            int stepStart = 0;
            if (trainState.ContainsKey("step"))
            {
                stepStart = trainState["step"].asscalar<int>();
            }

            // train loop
            for (int step = stepStart; step < 300; step++)
            {
                (NDarray data, NDarray label) = GetDataset(batchSize, weights);
                // predict := the output from the feedforward process
                NDarray predict = layer.FeedForward(data);

                // loss = 0.5 * (solution - predict) ^ 2
                NDarray loss = np.asarray(0.5) * np.mean(np.power(label - np.squeeze(predict), np.asarray(2)));
                // d_loss/d_predict = - (solution - predict)
                NDarray lossResultGradient = -np.reshape((label - np.squeeze(predict)), (batchSize, 1)) / batchSize;

                layer.BackPropagate(lossResultGradient);

                if (step % 10 == 0)
                {
                    Console.WriteLine($"Step: {step} | Loss: {loss.asscalar<double>().ToString("0.0000")}");
                    // save states
                    trainState["step"] = np.asarray(step);
                    SaveState(statePath, trainState);
                    layer.Save(statePath);
                }
            }
        }
    }
}
