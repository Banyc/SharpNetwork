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
            Scenario scenario = new Classification();
            // Scenario scenario = new Regression();

            ManualResetEvent stopTrainingSignal = new ManualResetEvent(false);

            Console.WriteLine("Start training...");
            Task trainingTask = new Task(() => scenario.Train(stopTrainingSignal));
            trainingTask.ContinueWith(_ => Console.WriteLine("Done training."));
            trainingTask.Start();

            // safely quit program
            Console.ReadLine();
            Console.WriteLine("Stopping training...");
            stopTrainingSignal.Set();
            trainingTask.Wait();
        }
    }
}
