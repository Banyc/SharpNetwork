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
            // this.batchSize = 128;

            // initialize DatasetLoader trainDataset here
            DatasetLoaderFactory datasetLoaderFactory = new DatasetLoaderFactory();

            this.trainDataset = datasetLoaderFactory.GetMnist(this.batchSize);
            // this.trainDataset = datasetLoaderFactory.GetCifar10(this.batchSize);
            // this.testDataset = datasetLoaderFactory.GetCifar10Test(this.batchSize / 10);

            OptimizerFactory optimizerFactory = new OptimizerFactory()
            {
                Type = OptimizerType.StochasticGradientDescent,
                LearningRate = 0.001,
            };

            // this.layers = new ImageLinearLayers(
            // this.layers = new Cnn2(
            this.layers = new Cnn2Fast(
            // this.layers = new Cnn1Fast(
                this.trainDataset.Dataset.Height,
                this.trainDataset.Dataset.Width,
                this.trainDataset.Dataset.Channels,
                this.trainDataset.Dataset.ClassNames.Count,
                optimizerFactory);

            this.StateFolderPath = $"trainings/{this.Name}/{this.layers.Name}";
        }

        public override void Train(ManualResetEvent stopTrainingSignal)
        {
            // loss function
            ILossFunction crossEntropy = new CrossEntropy()
            {
                LabelType = CrossEntropyLabelType.Embedded
            };
            Train(layers, crossEntropy, this.trainDataset, this.testDataset, 5, stopTrainingSignal);
        }
    }
}
