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

            BatchGradientDescent optimizer = new BatchGradientDescent()
            {
                // LearningRate = np.asarray(0.0001)
                LearningRate = np.asarray(0.001)
                // LearningRate = np.asarray(0.01)
            };

            // this.layers = new ImageLinearLayers(28, 28, 1, 10, optimizer);
            // this.layers = new Cnn2(28, 28, 1, 10, optimizer);
            // this.layers = new Cnn2Fast(28, 28, 1, 10, optimizer);
            this.layers = new Cnn1Fast(28, 28, 1, 10, optimizer);

            this.StateFolderPath = $"trainings/{this.Name}/{this.layers.Name}";
        }

        public override void Train(ManualResetEvent stopTrainingSignal)
        {
            // loss function
            ILossFunction crossEntropy = new CrossEntropy()
            {
                LabelType = CrossEntropyLabelType.Embedded
            };
            Train(layers, crossEntropy, this.trainDataset, 5, stopTrainingSignal);
        }
    }
}
