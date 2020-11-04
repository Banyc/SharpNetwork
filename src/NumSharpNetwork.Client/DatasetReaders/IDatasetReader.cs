using System.Collections.Generic;
using Numpy;

namespace NumSharpNetwork.Client.DatasetReaders
{
    public interface IDatasetReader
    {
        List<string> ClassNames { get; set; }
        // (dataPath, classIndex)
        List<(string, int)> Dataset { get; set; }
        (NDarray data, int classIndex) GetImageLabelPair(int datasetIndex);
        int Channels { get; set; }
        int Height { get; set; }
        int Width { get; set; }
    }
}
