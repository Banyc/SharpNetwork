using System;
using System.Collections.Generic;
using System.IO;
using System.Threading;
using Numpy;
using NumSharpNetwork.Shared.LossFunctions;
using NumSharpNetwork.Shared.Networks;
using NumSharpNetwork.Shared.Networks.Wrappers;

namespace NumSharpNetwork.Client.Scenarios
{
    public class Regression : Scenario
    {
        private NDarray weights;
        private int batchSize;
        private ILayer layers;

        public Regression()
        {
            this.Name = "Regression-1";
            // a hidden set of weights that is used to generate dataset
            this.weights = np.random.randn(10);
            this.batchSize = 200;

            this.layers = new ThreeLinearLayers();

            this.StateFolderPath = $"trainings/{this.Name}/{this.layers.Name}";
        }

        public override void Train(ManualResetEvent stopTrainingSignal)
        {
            Train(layers, stopTrainingSignal);
        }

        static (NDarray data, NDarray label) GetDataset(int batchSize, NDarray weights)
        {
            NDarray data = np.random.randn(batchSize, 10);
            NDarray label = np.sum(weights * data, 1);
            return (data, label);
        }

        private void Train(ILayer layer, ManualResetEvent stopTrainingSignal)
        {
            // loss function
            ILossFunction mse = new MeanSquaredError();

            // load state
            Dictionary<string, NDarray> trainState = LoadState();
            if (trainState.ContainsKey("weights"))
            {
                weights = trainState["weights"];
            }
            else
            {
                trainState["weights"] = weights;
            }
            layer.Load(this.StateFolderPath);

            // restore step number
            int stepStart = 0;
            if (trainState.ContainsKey("step"))
            {
                stepStart = trainState["step"].asscalar<int>();
            }

            // train loop
            for (int step = stepStart; step < 300; step++)
            {
                if (stopTrainingSignal.WaitOne(0))
                {
                    return;
                }
                (NDarray data, NDarray label) = GetDataset(batchSize, weights);
                // predict := the output from the feedforward process
                NDarray predict = layer.FeedForward(data);

                NDarray loss = mse.GetLoss(predict, label);
                NDarray lossResultGradient = mse.GetLossResultGradient(predict, label);

                layer.BackPropagate(lossResultGradient);

                if (step % 10 == 0)
                {
                    Console.WriteLine($"Step: {step} | Loss: {loss.asscalar<double>().ToString("0.0000")}");
                    // save states
                    trainState["step"] = np.asarray(step);
                    SaveState(trainState);
                    layer.Save(this.StateFolderPath);
                }
            }
        }
    }
}
