using System.Threading.Tasks;
using System;
using System.Collections.Generic;
using System.IO;
using System.Threading;
using Numpy;
using NumSharpNetwork.Client.Scenarios;
using NumSharpNetwork.Shared.Networks;
using NumSharpNetwork.Shared.Optimizers;

namespace NumSharpNetwork.Client
{
    class Program
    {
        static void Main(string[] args)
        {
            ManualResetEvent stopTrainingSignal = new ManualResetEvent(false);

            // Regression regression = new Regression();
            // regression.Train();

            Classification classification = new Classification();

            Console.WriteLine("Start training...");
            Task trainingTask = Task.Run(() => classification.Train(stopTrainingSignal));

            // user control
            Console.ReadLine();
            Console.WriteLine("Stopping training...");
            stopTrainingSignal.Set();
            trainingTask.Wait();
        }
    }
}
