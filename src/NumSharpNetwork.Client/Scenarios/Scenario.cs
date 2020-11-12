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
        private int Epoch { get; set; }

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

        protected void Train(ILayer layer, ILossFunction lossFunction, DatasetLoader trainDataset, DatasetLoader validationDataset, int numEpochs, ManualResetEvent stopTrainingSignal)
        {
            // print layer info
            Console.WriteLine("Layer info {");
            Console.WriteLine(layer);
            Console.WriteLine("} Layer info");

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
            for (this.Epoch = epochStart; this.Epoch < numEpochs; this.Epoch++)
            {
                if (stopTrainingSignal.WaitOne(0))
                {
                    return;
                }
                {
                    // train
                    layer.IsTrainMode = true;
                    ProceedOneEpoch(layer, stepStart, trainDataset, lossFunction, trainState, isSaveEveryStep: true, isBackpropagate: true, "Train", stopTrainingSignal);
                }
                if (validationDataset != null)
                {
                    // validate
                    layer.IsTrainMode = false;
                    ProceedOneEpoch(layer, 0, validationDataset, lossFunction, null, isSaveEveryStep: false, isBackpropagate: false, "Validation", stopTrainingSignal);
                }
                // reset step to 0 on the new epoch
                stepStart = 0;
                // save epoch number
                trainState["epoch"] = np.asarray(this.Epoch);
                SaveState(trainState);
            }
        }

        protected void ProceedOneEpoch(ILayer layer, int stepStart, DatasetLoader trainDataset, ILossFunction lossFunction, Dictionary<string, NDarray> trainState, bool isSaveEveryStep, bool isBackpropagate, string messagePrefix, ManualResetEvent stopTrainingSignal)
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
                if (isBackpropagate)
                {
                    NDarray lossResultGradient = lossFunction.GetLossResultGradient(predict, label);

                    // backpropagation to update weights and biases
                    layer.BackPropagate(lossResultGradient);
                }

                // print info
                if (step % 10 == 0)
                {
                    // get accuracy
                    double accuracy = GetAccuracy(trainDataset.BatchSize, predict, label);
                    // get loss
                    double meanRunningLoss = runningLoss / (step - stepStart + 1);
                    // print
                    Console.WriteLine($"[{messagePrefix}] Epoch: {this.Epoch} | Step: {step} | Loss: {meanRunningLoss:0.0000} | InstantLoss: {meanLoss:0.0000} | InstantAccuracy: {accuracy:0.000}");
                    // save states
                    if (trainState != null)
                    {
                        trainState["step"] = np.asarray(step);
                    }
                    if (isSaveEveryStep)
                    {
                        if (trainState != null)
                        {
                            SaveState(trainState);
                        }
                        layer.Save(this.StateFolderPath);
                    }
                }
                step++;
            }
        }

        protected void Validate(ILayer layer, DatasetLoader validationDataset, ILossFunction lossFunction, ManualResetEvent stopTrainingSignal)
        {
            foreach ((NDarray data, NDarray label) in validationDataset.GetBatches(isRandom: false))
            {
                if (stopTrainingSignal.WaitOne(0))
                {
                    return;
                }
                NDarray predict = layer.FeedForward(data);
                // get accuracy
                double accuracy = GetAccuracy(validationDataset.BatchSize, predict, label);
                // get loss
                NDarray loss = lossFunction.GetLoss(predict, label);
                double meanLoss = loss.mean();
                // print
                Console.WriteLine($"[Validation] InstantLoss: {meanLoss:0.0000} | InstantAccuracy: {accuracy:0.000}");
            }
        }

        private double GetAccuracy(int batchSize, NDarray predict, NDarray label)
        {
            NDarray maxPredict = np.argmax(predict, 1);
            NDarray trueMap = np.equal(maxPredict.reshape(batchSize, 1), label);
            double numTrue = trueMap.sum().asscalar<double>();
            int numSamples = batchSize;
            double accuracy = numTrue / numSamples;
            return accuracy;
        }
    }
}
