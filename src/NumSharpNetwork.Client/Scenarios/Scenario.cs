using System.Collections.Generic;
using System.IO;
using System.Threading;
using Numpy;

namespace NumSharpNetwork.Client.Scenarios
{
    public abstract class Scenario
    {
        public string Name { get; set; }
        public string StateFolderPath { get; set; }

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
    }
}
