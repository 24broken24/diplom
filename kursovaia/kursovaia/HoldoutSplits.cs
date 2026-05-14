using System;
using System.Collections.Generic;
using System.Linq;

namespace SymbolicRegression
{
    /// <summary>
    /// План внешнего разбиения (multi-var / 1-var в Program): train/test для сравнения кандидатов.
    /// Не связан с внутренним 80/20 в <see cref="SymbolicRegressionEngine.Evolve"/>.
    /// </summary>
    internal sealed class ValidationSchedule
    {
        public bool KFoldMode;
        public int NumRuns;
        public int TestCountHoldout;
        public double TestFraction;
        public int[]? Perm;
        public int[]? FoldStart;
        public int[]? FoldLen;
    }

    internal static class HoldoutSplits
    {
        internal static double MedianDoubles(List<double> vals)
        {
            if (vals.Count == 0)
                return double.NaN;
            var s = vals.OrderBy(v => v).ToArray();
            int m = s.Length / 2;
            return (s.Length & 1) == 1 ? s[m] : 0.5 * (s[m - 1] + s[m]);
        }

        internal static double QuantileLinear(double[] sorted, double p)
        {
            if (sorted.Length == 0) return double.NaN;
            if (sorted.Length == 1) return sorted[0];
            double pos = p * (sorted.Length - 1);
            int lo = (int)Math.Floor(pos);
            int hi = (int)Math.Ceiling(pos);
            double w = pos - lo;
            return sorted[lo] * (1 - w) + sorted[hi] * w;
        }

        internal static double IqrDoubles(List<double> vals)
        {
            if (vals.Count < 2)
                return 0;
            var s = vals.OrderBy(v => v).ToArray();
            return QuantileLinear(s, 0.75) - QuantileLinear(s, 0.25);
        }

        internal static bool IsKFoldSplitMode(string? mode) =>
            !string.IsNullOrWhiteSpace(mode)
            && (mode.Trim().Equals("kfold", StringComparison.OrdinalIgnoreCase)
                || mode.Trim().Equals("k-fold", StringComparison.OrdinalIgnoreCase));

        internal static int CandidateKindTieRank(string name) => name switch
        {
            "linear" => 0,
            "quadratic" => 1,
            "log-linear" => 2,
            _ => 3
        };

        internal static ValidationSchedule CreateValidationSchedule(MultiVarValidationConfig? mv, int n)
        {
            var s = new ValidationSchedule
            {
                KFoldMode = IsKFoldSplitMode(mv?.SplitMode),
                TestFraction = Math.Clamp(mv?.TestFraction ?? 0.2, 0.05, 0.49),
            };
            s.TestCountHoldout = Math.Max(1, (int)Math.Round(n * s.TestFraction));

            if (s.KFoldMode)
            {
                int k = mv?.KFolds ?? mv?.Repeats ?? 5;
                k = Math.Clamp(k, 2, Math.Min(40, Math.Max(2, n)));
                s.NumRuns = k;
                s.Perm = Enumerable.Range(0, n).ToArray();
                int shuffleSeed = mv?.ShuffleSeed ?? 12345;
                var rngPerm = new Random(SymbolicRegressionEngine.GpRandomizeBaseEachRun
                    ? Random.Shared.Next()
                    : shuffleSeed);
                for (int ii = n - 1; ii > 0; ii--)
                {
                    int jj = rngPerm.Next(ii + 1);
                    (s.Perm[ii], s.Perm[jj]) = (s.Perm[jj], s.Perm[ii]);
                }

                s.FoldLen = new int[k];
                int baseSz = n / k;
                int extra = n % k;
                for (int kk = 0; kk < k; kk++)
                    s.FoldLen![kk] = baseSz + (kk < extra ? 1 : 0);
                s.FoldStart = new int[k];
                s.FoldStart[0] = 0;
                for (int kk = 1; kk < k; kk++)
                    s.FoldStart[kk] = s.FoldStart[kk - 1] + s.FoldLen[kk - 1];
            }
            else
            {
                s.NumRuns = Math.Clamp(mv?.Repeats ?? 1, 1, 40);
            }

            return s;
        }

        internal static void GetTrainTestIndexLists(ValidationSchedule sch, int n, int rep, out int[] trainIdx, out int[] testIdx)
        {
            if (sch.KFoldMode)
            {
                int t0 = sch.FoldStart![rep];
                int tLen = sch.FoldLen![rep];
                var tr = new List<int>(n - tLen);
                for (int i = 0; i < t0; i++)
                    tr.Add(sch.Perm![i]);
                for (int i = t0 + tLen; i < n; i++)
                    tr.Add(sch.Perm[i]);
                trainIdx = tr.ToArray();
                testIdx = new int[tLen];
                for (int i = 0; i < tLen; i++)
                    testIdx[i] = sch.Perm[t0 + i];
            }
            else
            {
                var splitRng = new Random(SymbolicRegressionEngine.GpRandomizeBaseEachRun
                    ? Random.Shared.Next()
                    : unchecked((int)(123 + (long)rep * 9973)));
                var indices = Enumerable.Range(0, n).ToArray();
                for (int ii = n - 1; ii > 0; ii--)
                {
                    int jj = splitRng.Next(ii + 1);
                    (indices[ii], indices[jj]) = (indices[jj], indices[ii]);
                }
                int lastTrainCount = n - sch.TestCountHoldout;
                trainIdx = new int[lastTrainCount];
                for (int i = 0; i < lastTrainCount; i++)
                    trainIdx[i] = indices[i];
                testIdx = new int[sch.TestCountHoldout];
                for (int i = 0; i < sch.TestCountHoldout; i++)
                    testIdx[i] = indices[lastTrainCount + i];
            }
        }
    }
}
