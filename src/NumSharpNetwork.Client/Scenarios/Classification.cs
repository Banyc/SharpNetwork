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
        private string Name { get; set; } = "Classification-1";

        public Classification()
        {
            this.batchSize = 16;

            // initialize DatasetLoader trainDataset here
            DatasetLoaderFactory datasetLoaderFactory = new DatasetLoaderFactory();

            this.trainDataset = datasetLoaderFactory.GetMnist(this.batchSize);

            BasicOptimizer optimizer = new BasicOptimizer()
            {
                // LearningRate = np.asarray(0.0001)
                LearningRate = np.asarray(0.001)
                // LearningRate = np.asarray(0.01)
            };

            this.layers = new ImageLinearLayers(28, 28, 1, 10, optimizer);
            // this.layers = new Cnn(28, 28, 1, 10, optimizer);

            this.stateFolderPath = "trainings/classification";
        }

        public void Train()
        {
            Train(layers, this.trainDataset, stateFolderPath, this.Name, 5);
        }

        static Dictionary<string, NDarray> LoadState(string stateFolderPath, string fileNamePrefix)
        {
            Directory.CreateDirectory(stateFolderPath);
            string statePath = Path.Combine(stateFolderPath, fileNamePrefix);
            Dictionary<string, NDarray> stateDict = new Dictionary<string, NDarray>();

            string[] filePaths = Directory.GetFiles(stateFolderPath);
            foreach (string filePath in filePaths)
            {
                string baseName = Path.GetFileNameWithoutExtension(filePath);
                if (baseName.Contains(fileNamePrefix))
                {
                    string postFix = baseName[(fileNamePrefix.Length + 1)..];
                    stateDict[postFix] = np.load($"{statePath}.{postFix}.npy");
                }
            }
            return stateDict;
        }

        static void SaveState(string stateFolderPath, Dictionary<string, NDarray> state, string fileNamePrefix)
        {
            Directory.CreateDirectory(stateFolderPath);
            string statePath = Path.Combine(stateFolderPath, fileNamePrefix);
            foreach ((string key, NDarray value) in state)
            {
                np.save($"{statePath}.{key}.npy", value);
            }
        }

        static void Train(ILayer layer, DatasetLoader trainDataset, string statePath, string stateFilenamePrefix, int numEpoches)
        {
            // loss function
            ILossFunction crossEntropy = new CrossEntropy()
            {
                LabelType = CrossEntropyLabelType.Embedded
            };

            // load state
            Dictionary<string, NDarray> trainState = LoadState(statePath, stateFilenamePrefix);
            layer.Load(statePath);

            // restore step number
            int stepStart = 0;
            if (trainState.ContainsKey("step"))
            {
                stepStart = trainState["step"].asscalar<int>();
            }

            // restore epoch number
            int epochStart = 0;
            if (trainState.ContainsKey("epoch"))
            {
                epochStart = trainState["epoch"].asscalar<int>();
            }

            // train loop
            int epoch;
            for (epoch = epochStart; epoch < numEpoches; epoch++)
            {
                TrainOneEpoch(layer, stepStart, trainDataset, crossEntropy, trainState, statePath, stateFilenamePrefix);
                // reset step to 0 on the new epoch
                stepStart = 0;
                // save epoch number
                trainState["epoch"] = np.asarray(epoch);
                SaveState(statePath, trainState, stateFilenamePrefix);
            }
        }

        static void TrainOneEpoch(ILayer layer, int stepStart, DatasetLoader trainDataset, ILossFunction lossFunction, Dictionary<string, NDarray> trainState, string statePath, string stateFilenamePrefix)
        {
            // // DEBUG ONLY
            // double previousLoss = -1;

            // train loop
            double runningLoss = 0;
            int step = stepStart;
            foreach ((NDarray data, NDarray label) in trainDataset.GetBatches(step))
            {
                // // test
                // NDarray data = np.ones(batchSize, 1, 28, 28);
                // NDarray label = np.ones(batchSize, 1);

                // predict := the output from the feedforward process
                NDarray predict = layer.FeedForward(data);

                NDarray loss = lossFunction.GetLoss(predict, label);
                double meanLoss = loss.mean();
                runningLoss += meanLoss;
                NDarray lossResultGradient = lossFunction.GetLossResultGradient(predict, label);

                // // DEBUG ONLY
                // if (previousLoss != -1 && (loss.mean() > previousLoss + 2))
                // {
                //     throw new Exception("Sudden Climbing detected");
                // }
                // previousLoss = loss.mean();

                layer.BackPropagate(lossResultGradient);

                if (step % 10 == 0)
                {
                    double meanRunningLoss = runningLoss / (step + 1);
                    Console.WriteLine($"Step: {step} | Loss: {meanRunningLoss.ToString("0.0000")} | InstantLoss: {meanLoss.ToString("0.0000")}");
                    // save states
                    trainState["step"] = np.asarray(step);
                    SaveState(statePath, trainState, stateFilenamePrefix);
                    layer.Save(statePath);
                }
                step++;
            }
        }
    }
}
