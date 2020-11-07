using System;
using System.Collections.Generic;
using System.IO;
using Numpy;
using NumSharpNetwork.Shared.LossFunctions;
using NumSharpNetwork.Shared.Networks.Wrappers;
using NumSharpNetwork.Shared.Networks;
using NumSharpNetwork.Shared.Optimizers;
using NumSharpNetwork.Client.DatasetReaders;
using System.Threading;

namespace NumSharpNetwork.Client.Scenarios
{
    public class Classification : Scenario
    {
        private int batchSize;
        private ILayer layers;
        private DatasetLoader trainDataset;
        private DatasetLoader testDataset;

        public Classification()
        {
            this.Name = "Classification-1";

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

            // this.layers = new ImageLinearLayers(28, 28, 1, 10, optimizer);
            this.layers = new Cnn(28, 28, 1, 10, optimizer);

            this.StateFolderPath = $"trainings/{this.Name}/{this.layers.Name}";
        }

        public override void Train(ManualResetEvent stopTrainingSignal)
        {
            Train(layers, this.trainDataset, 5, stopTrainingSignal);
        }

        private void Train(ILayer layer, DatasetLoader trainDataset, int numEpoches, ManualResetEvent stopTrainingSignal)
        {
            // loss function
            ILossFunction crossEntropy = new CrossEntropy()
            {
                LabelType = CrossEntropyLabelType.Embedded
            };

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
            for (epoch = epochStart; epoch < numEpoches; epoch++)
            {
                if (stopTrainingSignal.WaitOne(0))
                {
                    return;
                }
                TrainOneEpoch(layer, stepStart, trainDataset, crossEntropy, trainState, stopTrainingSignal);
                // reset step to 0 on the new epoch
                stepStart = 0;
                // save epoch number
                trainState["epoch"] = np.asarray(epoch);
                SaveState(trainState);
            }
        }

        private void TrainOneEpoch(ILayer layer, int stepStart, DatasetLoader trainDataset, ILossFunction lossFunction, Dictionary<string, NDarray> trainState, ManualResetEvent stopTrainingSignal)
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
                    NDarray trueMap = np.equal(maxPredict.reshape(batchSize, 1), label);
                    double numTrue = trueMap.sum().asscalar<double>();
                    int numSamples = batchSize;
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
