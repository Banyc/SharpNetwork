using System.IO;
using System.Collections.Generic;
using Numpy;
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

            // ISSUE: PIL.image is not supported...
            byte[] dataBytes = GetDataBytes(dataPath);

            NDarray data = np.array(dataBytes).reshape(1, this.Channels, this.Height, this.Width);
            return (data, classIndex);
        }

        private byte[] GetDataBytes(string dataPath)
        {
            Image image = Image.FromFile(dataPath);
            Bitmap bitmap = new Bitmap(image);
            byte[] dataBytes = new byte[this.Channels * this.Height * this.Width];

            int pixelX;
            int pixelY;

            int redBaseIndex = 0;
            int blueBaseIndex = this.Height * this.Width;
            int greenBaseIndex = 2 * this.Height * this.Width;

            int indexOffset = 0;
            for (pixelY = 0; pixelY < bitmap.Height; pixelY++)
            {
                for (pixelX = 0; pixelX < bitmap.Height; pixelX++)
                {
                    var pixel = bitmap.GetPixel(pixelX, pixelY);
                    if (this.Channels == 3)
                    {
                        // three channels
                        dataBytes[redBaseIndex + indexOffset] = pixel.R;
                        dataBytes[blueBaseIndex + indexOffset] = pixel.B;
                        dataBytes[greenBaseIndex + indexOffset] = pixel.G;
                    }
                    else
                    {
                        // grayscale
                        dataBytes[indexOffset] = (byte)((0.3 * pixel.R) + (0.59 * pixel.G) + (0.11 * pixel.B));
                    }

                    indexOffset++;
                }
            }
            return dataBytes;
        }
    }
}
