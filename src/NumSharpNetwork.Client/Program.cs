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
            // change the training scenario here
            Scenario scenario = new Classification();
            // Scenario scenario = new Regression();

            // don't receive exception
            bool isDebug = false;
            // // receive exception
            // bool isDebug = true;

            ManualResetEvent stopTrainingSignal = new ManualResetEvent(false);

            Console.WriteLine("Start training...");
            if (isDebug)
            {
                scenario.Train(stopTrainingSignal);
            }
            else
            {
                Task trainingTask = new Task(() => scenario.Train(stopTrainingSignal));
                trainingTask.ContinueWith((t) =>
                {
                    if (t.IsFaulted) throw t.Exception;
                    if (t.IsCompleted) Console.WriteLine("Done training.");//optionally do some work);
                });
                trainingTask.Start();

                // safely quit program
                Console.ReadLine();
                Console.WriteLine("Stopping training...");
                stopTrainingSignal.Set();
                trainingTask.Wait();
            }
        }
    }
}
