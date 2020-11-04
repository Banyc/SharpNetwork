using System.IO;
using System.Collections.Generic;
using Numpy;
using SixLabors.ImageSharp;
using System.Drawing;

namespace NumSharpNetwork.Client.DatasetReaders
{
    public class GeneralDatasetReader : IDatasetReader
    {
        public string RootFolderPath { get; set; }
        public List<string> ClassNames { get; set; }
        // (dataPath, classIndex)
        public List<(string, int)> Dataset { get; set; }
        public int Channels { get; set; }
        public int Height { get; set; }
        public int Width { get; set; }

        public GeneralDatasetReader(string rootFolderPath, int channels, int height, int width)
        {
            this.RootFolderPath = rootFolderPath;
            this.Channels = channels;
            this.Height = height;
            this.Width = width;

            BuildIndex();
        }

        public void BuildIndex()
        {
            this.ClassNames = new List<string>();
            this.Dataset = new List<(string, int)>();

            string[] classPaths = Directory.GetDirectories(this.RootFolderPath);
            int classIndex;
            for (classIndex = 0; classIndex < classPaths.Length; classIndex++)
            {
                this.ClassNames.Add(Path.GetDirectoryName(classPaths[classIndex]));
                string[] dataPaths = Directory.GetFiles(classPaths[classIndex]);
                foreach (string dataPath in dataPaths)
                {
                    this.Dataset.Add((dataPath, classIndex));
                }
            }
        }

        public (NDarray data, int classIndex) GetImageLabelPair(int datasetIndex)
        {
            (string dataPath, int classIndex) = this.Dataset[datasetIndex];
            // ISSUE: PIL.image is not support...
            byte[] dataBytes = File.ReadAllBytes(dataPath);
            NDarray data = np.array(dataBytes).reshape(1, this.Channels, this.Height, this.Width);
            return (data, classIndex);
        }
    }
}
