using System;
using System.IO;
using Numpy;
using NumSharpNetwork.Shared.Optimizers;

namespace NumSharpNetwork.Shared.Networks
{
    public class BatchNormalizationRecord
    {
        public NDarray Gamma { get; set; }
        public NDarray Beta { get; set; }
        public NDarray Mean { get; set; }
        public NDarray Variance { get; set; }
        public NDarray StandardDerivation { get; set; }
        public NDarray StandardScore { get; set; }
        public NDarray Input { get; set; }
        public NDarray NormalizedInput { get; set; }
    }

    // Warning: this implementation might encounter overflow issuses
    public class BatchNormalization : ILayer
    {
        public string Name { get; set; } = "BatchNormalization";
        public bool IsTrainMode { get; set; } = true;
        // public int InputChannels { get; set; }
        private IOptimizer GammaOptimizer { get; set; }
        private IOptimizer BetaOptimizer { get; set; }
        public double Momentum { get; set; } = 0.9;
        public bool IsSpatial { get; set; } = false;

        // learnable parameter gamma and beta to restore the expression of the original dataset
        private NDarray Gamma { get; set; }
        private NDarray Beta { get; set; }

        private NDarray RunningMean { get; set; }
        private NDarray RunningVariance { get; set; }

        // in case variance being zero
        private double Epsilon { get; set; } = 0.00001;

        private BatchNormalizationRecord Record { get; set; } = new BatchNormalizationRecord();


        public BatchNormalization(int inputChannels, OptimizerFactory optimizerFactory, string name = "BatchNormalization")
        {
            // this.InputChannels = inputChannels;
            this.GammaOptimizer = optimizerFactory.GetOptimizer();
            this.BetaOptimizer = optimizerFactory.GetOptimizer();
            this.Name = name;

            this.Gamma = np.ones(inputChannels);
            this.Beta = np.zeros(inputChannels);

            this.RunningMean = np.zeros(inputChannels);
            this.RunningVariance = np.zeros(inputChannels);
        }

        public NDarray BackPropagate(NDarray lossResultGradient)
        {
            // compress the result
            int[] inputShape = lossResultGradient.shape.Dimensions;
            if (this.IsSpatial)
            {
                lossResultGradient = CompressInput(lossResultGradient);
            }

            NDarray dx = BackPropagateZllz4(lossResultGradient);
            NDarray lossInputGradient = dx;

            // NDarray lossBetaGradient = lossResultGradient.sum(0);
            // NDarray lossGammaGradient = (lossResultGradient * this.Record.StandardScore).sum(0);

            // // d_loss / d_x
            // int batchSize = lossResultGradient.shape[0];
            // double meanInputGradient = 1.0 / batchSize;
            // double varianceInputGradient = (2.0 / batchSize) * (1 - meanInputGradient);
            // NDarray resultInputGradient =
            //     this.Gamma *
            //     (
            //         (
            //             (this.Record.Input - this.Record.Mean) *
            //             (-0.5 * Math.Pow(varianceInputGradient - this.Epsilon, -3 / 2))
            //         ) +
            //         (
            //             (1 - meanInputGradient) *
            //             (
            //                 1.0 / np.sqrt(this.Record.Variance - this.Epsilon)
            //             )
            //         )
            //     );
            // NDarray lossInputGradient = lossResultGradient * resultInputGradient;

            // // // DEBUG
            // // if (!double.IsFinite(meanInputGradient))
            // // {
            // //     throw new Exception();
            // // }
            // // if (double.IsNaN(meanInputGradient))
            // // {
            // //     throw new Exception();
            // // }
            // // if (!double.IsFinite(varianceInputGradient))
            // // {
            // //     throw new Exception();
            // // }
            // // if (double.IsNaN(varianceInputGradient))
            // // {
            // //     throw new Exception();
            // // }
            // // if (np.isnan(resultInputGradient).any())
            // // {
            // //     throw new Exception();
            // // }
            // // if (np.isinf(resultInputGradient).any())
            // // {
            // //     throw new Exception();
            // // }
            // // if (!np.equal(dx, lossInputGradient).all())
            // // {
            // //     throw new Exception();
            // // }

            // this.Beta = this.Optimizer.Optimize(this.Beta, lossBetaGradient);
            // this.Gamma = this.Optimizer.Optimize(this.Gamma, lossGammaGradient);

            // restore the input
            if (this.IsSpatial)
            {
                lossInputGradient = ExpandInput(lossInputGradient, inputShape);
            }

            return lossInputGradient;
        }

        public NDarray FeedForward(NDarray input)
        {
            // compress the input
            int[] inputShape = input.shape.Dimensions;
            if (this.IsSpatial)
            {
                input = CompressInput(input);
            }
            // standard score
            NDarray mean = input.mean(0);
            NDarray variance = input.var(0);
            NDarray standardDerivation = np.sqrt(variance + this.Epsilon);
            NDarray standardScore = (input - mean.reshape(1, -1)) / standardDerivation.reshape(1, -1);

            // batch normalization
            NDarray batchNormalizedInput = standardScore * this.Gamma.reshape(1, -1) + this.Beta.reshape(1, -1);

            // save
            this.Record.Beta = this.Beta;
            this.Record.Gamma = this.Gamma;
            this.Record.Input = input;
            this.Record.NormalizedInput = batchNormalizedInput;
            this.Record.Mean = mean;
            this.Record.Variance = variance;
            this.Record.StandardDerivation = standardDerivation;
            this.Record.StandardScore = standardScore;

            this.RunningMean = mean * (1 - this.Momentum) + this.RunningMean * this.Momentum;
            this.RunningVariance = variance * (1 - this.Momentum) + this.RunningVariance * this.Momentum;

            // restore the result
            if (this.IsSpatial)
            {
                batchNormalizedInput = ExpandInput(batchNormalizedInput, inputShape);
            }

            return batchNormalizedInput;
        }

        public void Load(string folderPath)
        {
            Directory.CreateDirectory(folderPath);
            string statePath = Path.Combine(folderPath, $"{this.Name}.npy");
            string gammaPath = $"{statePath}.gamma.npy";
            string betaPath = $"{statePath}.beta.npy";
            if (File.Exists(gammaPath))
            {
                this.Gamma = np.load(gammaPath);
            }
            if (File.Exists(betaPath))
            {
                this.Beta = np.load(betaPath);
            }
        }

        public void Save(string folderPath)
        {
            Directory.CreateDirectory(folderPath);
            string statePath = Path.Combine(folderPath, $"{this.Name}.npy");
            np.save($"{statePath}.gamma.npy", this.Gamma);
            np.save($"{statePath}.beta.npy", this.Beta);
        }

        // input.shape = [batchSize, inputChannels, height, width]
        // return shape = [batchSize * height * width, inputChannels]
        private NDarray CompressInput(NDarray input)
        {
            int batchSize = input.shape[0];
            int inputChannels = input.shape[1];
            // transposed.shape = [batchSize, height, width, inputChannels]
            NDarray transposed = input.transpose(0, 2, 3, 1);
            return transposed.reshape(-1, inputChannels);
        }

        private NDarray ExpandInput(NDarray compressedInput, int[] inputShape)
        {
            int batchSize = inputShape[0];
            int inputChannels = inputShape[1];
            int height = inputShape[2];
            int width = inputShape[3];

            NDarray transposed = compressedInput.reshape(batchSize, height, width, inputChannels);
            NDarray expandedInput = transposed.transpose(0, 3, 1, 2);
            return expandedInput;
        }

        private NDarray BackPropagateZllz4(NDarray dout)
        {
            var x = this.Record.Input;
            var gamma = this.Record.Gamma;
            var beta = this.Record.Beta;
            var mean = this.Record.Mean;
            var variance = this.Record.Variance;
            var x_hat = this.Record.StandardScore;
            var eps = this.Epsilon;

            // gamma 的梯度
            var dgamma = np.sum(dout * x_hat, 0);
            // beta 的梯度
            var dbeta =  np.sum(dout, 0);
            // x_hat 对 dout 的梯度
            var dx_hat = dout * gamma.reshape((1,-1));
            // x 对 x_hat 的梯度
            var dxi_of_x_hat = dx_hat / np.power( (variance.reshape((1,-1)) + eps), np.asarray(0.5));
            // mean 对 x_hat 的梯度
            var dmean_of_x_hat = np.sum(-dx_hat / np.power( (variance.reshape((1,-1)) + eps), np.asarray(0.5)), 0) ;
            // var 对 x_hat 的梯度
            var dvar_of_x_hat = np.sum(dx_hat * (-1/2) * (np.power((variance.reshape((1,-1)) + eps), np.asarray(-1.5))) * (x - mean.reshape((1, -1))), 0);
            // x 对 mean 的梯度
            var dxi_of_mean = 1/x.shape[0] * dmean_of_x_hat * np.ones_like(x);
            // x 对 var 的梯度
            var dxi_of_var = 1/x.shape[0] * 2 * (x-mean.reshape((1,-1))) * dvar_of_x_hat.reshape((1,-1))  ;
            // x 对 dout 的梯度，三个加起来
            var dxi = dxi_of_x_hat + dxi_of_mean + dxi_of_var;
            // 得到 dx
            var dx = dxi;

            // 更新参数
            // self.gamma = self.optimizer.optim(self.gamma, dgamma)
            // self.beta = self.optimizer.optim(self.beta, dbeta)

            this.Beta = this.BetaOptimizer.Optimize(this.Beta, dbeta);
            this.Gamma = this.GammaOptimizer.Optimize(this.Gamma, dgamma);

            return dx;
        }

        public override string ToString()
        {
            return $"{this.Name} [{this.Gamma.shape[0]}]";
        }
    }
}
