using System;
using System.Collections.Generic;
using System.IO;
using NumSharp;
using NumSharpNetwork.Shared.Networks;

namespace NumSharpNetwork.Client
{
    class Program
    {
        static void Main(string[] args)
        {
            NDArray weights = np.random.randn(10);
            int batchSize = 200;

            ThreeLinearLayers layers = new ThreeLinearLayers();

            const string statePath = "trainings";
            Train(layers, batchSize, weights, statePath);
        }

        static Dictionary<string, Array> LoadState(string folderPath)
        {
            Directory.CreateDirectory(folderPath);
            string statePath = Path.Combine(folderPath, $"test1.npy");
            if (!File.Exists(statePath))
            {
                return new Dictionary<string, Array>();
            }
            NpzDictionary<Array> loadedState = np.Load_Npz<Array>(statePath);
            return new Dictionary<string, Array>(loadedState);
        }

        static void SaveState(string folderPath, Dictionary<string, Array> state)
        {
            Directory.CreateDirectory(folderPath);
            string statePath = Path.Combine(folderPath, $"test1.npy");
            np.Save_Npz(state, statePath);
        }

        static (NDArray data, NDArray label) GetData(int batchSize, NDArray weights)
        {
            NDArray data = np.random.randn(batchSize, 10);
            NDArray label = np.sum(weights * data, 1);
            return (data, label);
        }

        static void Train(ILayer layer, int batchSize, NDArray weights, string statePath)
        {
            Dictionary<string, Array> trainState = LoadState(statePath);
            if (trainState.ContainsKey("weights"))
            {
                weights = trainState["weights"];
            }
            else
            {
                trainState["weights"] = (Array)weights;
            }
            layer.Load(statePath);

            int stepStart = 0;
            if (trainState.ContainsKey("step"))
            {
                stepStart = ((NDArray)trainState["step"])[0];
            }

            for (int step = stepStart; step < 300; step++)
            {
                (NDArray data, NDArray label) = GetData(batchSize, weights);
                NDArray predict = layer.Forward(data);

                NDArray loss = 0.5 * np.mean(np.power(label - np.squeeze(predict), 2));
                NDArray lossResultGradient = -np.reshape((label - np.squeeze(predict)), (batchSize, 1)) / batchSize;
                Console.WriteLine($"Step: {step} | Loss: {loss}");

                layer.Backward(lossResultGradient);

                trainState["step"] = (Array)new NDArray(new int[] { step });
                SaveState(statePath, trainState);
                layer.Save(statePath);
            }
        }
    }
}
