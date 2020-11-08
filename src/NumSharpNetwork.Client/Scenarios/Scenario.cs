using System;
using System.Collections.Generic;
using System.IO;
using System.Threading;
using Numpy;
using NumSharpNetwork.Client.DatasetReaders;
using NumSharpNetwork.Shared.LossFunctions;
using NumSharpNetwork.Shared.Networks;

namespace NumSharpNetwork.Client.Scenarios
{
    public abstract class Scenario
    {
        public string Name { get; set; }
        public string StateFolderPath { get; set; }

        public abstract void Train(ManualResetEvent stopTrainingSignal);

        protected Dictionary<string, NDarray> LoadState()
        {
            Directory.CreateDirectory(this.StateFolderPath);
            string statePath = Path.Combine(this.StateFolderPath, this.Name);
            Dictionary<string, NDarray> stateDict = new Dictionary<string, NDarray>();

            string[] filePaths = Directory.GetFiles(this.StateFolderPath);
            foreach (string filePath in filePaths)
            {
                string baseName = Path.GetFileNameWithoutExtension(filePath);
                if (baseName.Contains(this.Name))
                {
                    string postFix = baseName[(this.Name.Length + 1)..];
                    stateDict[postFix] = np.load($"{statePath}.{postFix}.npy");
                }
            }
            return stateDict;
        }

        protected void SaveState(Dictionary<string, NDarray> state)
        {
            Directory.CreateDirectory(this.StateFolderPath);
            string statePath = Path.Combine(this.StateFolderPath, this.Name);
            foreach ((string key, NDarray value) in state)
            {
                np.save($"{statePath}.{key}.npy", value);
            }
        }

        protected void Train(ILayer layer, ILossFunction lossFunction, DatasetLoader trainDataset, int numEpochs, ManualResetEvent stopTrainingSignal)
        {
            // load state
            Dictionary<string, NDarray> trainState = LoadState();
            layer.Load(this.StateFolderPath);

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
            for (epoch = epochStart; epoch < numEpochs; epoch++)
            {
                if (stopTrainingSignal.WaitOne(0))
                {
                    return;
                }
                TrainOneEpoch(layer, stepStart, trainDataset, lossFunction, trainState, stopTrainingSignal);
                // reset step to 0 on the new epoch
                stepStart = 0;
                // save epoch number
                trainState["epoch"] = np.asarray(epoch);
                SaveState(trainState);
            }
        }

        protected void TrainOneEpoch(ILayer layer, int stepStart, DatasetLoader trainDataset, ILossFunction lossFunction, Dictionary<string, NDarray> trainState, ManualResetEvent stopTrainingSignal)
        {
            // // DEBUG ONLY
            // double previousLoss = -1;

            // train loop
            double runningLoss = 0;
            int step = stepStart;
            foreach ((NDarray data, NDarray label) in trainDataset.GetBatches(step))
            {
                if (stopTrainingSignal.WaitOne(0))
                {
                    return;
                }
                // // test
                // NDarray data = np.ones(batchSize, 1, 28, 28);
                // NDarray label = np.ones(batchSize, 1);

                // predict := the output from the feedforward process
                NDarray predict = layer.FeedForward(data);

                // loss function takes effect
                NDarray loss = lossFunction.GetLoss(predict, label);
                double meanLoss = loss.mean();
                runningLoss += meanLoss;
                NDarray lossResultGradient = lossFunction.GetLossResultGradient(predict, label);

                // backpropagation to update weights and biases
                layer.BackPropagate(lossResultGradient);

                // print info
                if (step % 10 == 0)
                {
                    // get accuracy
                    NDarray maxPredict = np.argmax(predict, 1);
                    NDarray trueMap = np.equal(maxPredict.reshape(trainDataset.BatchSize, 1), label);
                    double numTrue = trueMap.sum().asscalar<double>();
                    int numSamples = trainDataset.BatchSize;
                    double accuracy = numTrue / numSamples;
                    // get loss
                    double meanRunningLoss = runningLoss / (step + 1);
                    // print
                    Console.WriteLine($"Step: {step} | Loss: {meanRunningLoss.ToString("0.0000")} | InstantLoss: {meanLoss.ToString("0.0000")} | InstantAccuracy: {accuracy.ToString("0.000")}");
                    // save states
                    trainState["step"] = np.asarray(step);
                    SaveState(trainState);
                    layer.Save(this.StateFolderPath);
                }
                step++;
            }
        }
    }
}
