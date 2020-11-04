using System.Collections.Generic;
using Numpy;

namespace NumSharpNetwork.Client.DatasetReaders
{
    public class DatasetLoader
    {
        public IDatasetReader Dataset { get; set; }
        public int BatchSize { get; set; }
        public DatasetLoader(IDatasetReader dataset, int batchSize)
        {
            this.Dataset = dataset;
            this.BatchSize = batchSize;
        }

        public IEnumerable<(NDarray dataBatches, NDarray embeddedLabelBatches)> GetBatches(int startStep = 0)
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
                    (NDarray data, int classIndex) = this.Dataset.GetImageLabelPair(dataIndexOffset + baseDataIndex);
                    dataBatches[$"{dataIndexOffset}"] = data;
                    embeddedLabelBatches[$"{dataIndexOffset}"] = np.asarray(classIndex);
                }

                yield return (dataBatches, embeddedLabelBatches);
            }
        }
    }
}
