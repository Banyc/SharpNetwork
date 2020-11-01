using System;
using System.Collections.Generic;
using System.IO;
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
            Regression regression = new Regression();
            regression.Train();
        }
    }
}
