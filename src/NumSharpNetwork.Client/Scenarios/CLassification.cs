using System;
using System.Collections.Generic;
using System.IO;
using Numpy;
using NumSharpNetwork.Shared.LossFunctions;
using NumSharpNetwork.Shared.Networks.Wrappers;
using NumSharpNetwork.Shared.Networks;
using NumSharpNetwork.Shared.Optimizers;
using NumSharpNetwork.Client.DatasetReaders;

namespace NumSharpNetwork.Client.Scenarios
{
    public class Classification
    {
        private int batchSize;
        private ILayer layers;
        private string stateFolderPath;
        private DatasetLoader trainDataset;
        private DatasetLoader testDataset;

        public Classification()
        {
            this.batchSize = 2;

            // initialize DatasetLoader trainDataset here

            BasicOptimizer optimizer = new BasicOptimizer()
            {
                LearningRate = np.asarray(0.001)
            };

            this.layers = new Cnn(28, 28, 1, 10, optimizer);

            this.stateFolderPath = "trainings/classification";
        }

        public void Train()
        {
            Train(layers, batchSize, this.trainDataset, stateFolderPath);
        }

        static Dictionary<string, NDarray> LoadState(string stateFolderPath)
        {
            Directory.CreateDirectory(stateFolderPath);
            string statePath = Path.Combine(stateFolderPath, $"Classification-1.npy");
            Dictionary<string, NDarray> stateDict = new Dictionary<string, NDarray>();

            if (File.Exists($"{statePath}.step.npy"))
            {
                stateDict["step"] = np.load($"{statePath}.step.npy");
            }
            return stateDict;
        }

        static void SaveState(string stateFolderPath, Dictionary<string, NDarray> state)
        {
            Directory.CreateDirectory(stateFolderPath);
            string statePath = Path.Combine(stateFolderPath, $"Classification-1.npy");
            np.save($"{statePath}.step.npy", state["step"]);
        }

        static void Train(ILayer layer, int batchSize, DatasetLoader trainDataset, string statePath)
        {
            // loss function
            ILossFunction crossEntropy = new CrossEntropy()
            {
                LabelType = CrossEntropyLabelType.Embedded
            };

            // load state
            Dictionary<string, NDarray> trainState = LoadState(statePath);
            layer.Load(statePath);

            // restore step number
            int stepStart = 0;
            if (trainState.ContainsKey("step"))
            {
                stepStart = trainState["step"].asscalar<int>();
            }

            // train loop
            int step = stepStart;
            // foreach ((NDarray data, NDarray label) in trainDataset.GetBatches(step))
            {
                // test
                NDarray data = np.ones(batchSize, 1, 28, 28);
                NDarray label = np.ones(batchSize, 1);

                // predict := the output from the feedforward process
                NDarray predict = layer.FeedForward(data);

                NDarray loss = crossEntropy.GetLoss(predict, label);
                NDarray lossResultGradient = crossEntropy.GetLossResultGradient(predict, label);

                layer.BackPropagate(lossResultGradient);

                if (step % 10 == 0)
                {
                    Console.WriteLine($"Step: {step} | Loss: {loss.mean().ToString("0.0000")}");
                    // save states
                    trainState["step"] = np.asarray(step);
                    SaveState(statePath, trainState);
                    layer.Save(statePath);
                }
                step++;
            }
        }
    }
}
