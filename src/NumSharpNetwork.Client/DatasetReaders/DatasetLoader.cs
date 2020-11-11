using System;
using System.Collections.Generic;
using Numpy;

namespace NumSharpNetwork.Client.DatasetReaders
{
    // load Dataset from files
    public class DatasetLoader
    {
        public IDatasetReader Dataset { get; set; }
        public int BatchSize { get; set; }
        private Random Random { get; set; } = new Random();
        public DatasetLoader(IDatasetReader dataset, int batchSize)
        {
            this.Dataset = dataset;
            this.BatchSize = batchSize;
        }

        public IEnumerable<(NDarray dataBatches, NDarray embeddedLabelBatches)> GetBatches(int startStep = 0, bool isRandom = true)
        {
            int numBatches = this.Dataset.Dataset.Count / this.BatchSize;
            int batchIndex;
            for (batchIndex = startStep; batchIndex < numBatches; batchIndex++)
            {
                NDarray dataBatches = np.empty(this.BatchSize, this.Dataset.Channels, this.Dataset.Height, this.Dataset.Width);
                NDarray embeddedLabelBatches = np.empty(this.BatchSize, 1);
                int baseDataIndex = batchIndex * this.BatchSize;
                int dataIndexOffset;
                for (dataIndexOffset = 0; dataIndexOffset < this.BatchSize; dataIndexOffset++)
                {
                    NDarray data = null;
                    int classIndex = -1;
                    if (!isRandom)
                    {
                        (data, classIndex) = this.Dataset.GetImageLabelPair(dataIndexOffset + baseDataIndex);
                    }
                    else
                    {
                        int randomIndex = this.Random.Next(0, this.Dataset.Dataset.Count);
                        (data, classIndex) = this.Dataset.GetImageLabelPair(randomIndex);
                    }
                    dataBatches[$"{dataIndexOffset}"] = data;
                    embeddedLabelBatches[$"{dataIndexOffset}"] = np.asarray(classIndex);
                }

                yield return (dataBatches, embeddedLabelBatches);
            }
        }
    }
}
