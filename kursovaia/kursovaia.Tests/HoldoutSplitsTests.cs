using System;
using System.Collections.Generic;
using System.Linq;
using Xunit;

namespace SymbolicRegression.Tests;

/// <summary>
/// Проверки внешнего train/test из <see cref="HoldoutSplits"/> без Excel и GP.
/// Перед сценариями с детерминизмом выключаем <see cref="SymbolicRegressionEngine.GpRandomizeBaseEachRun"/>.
/// </summary>
public sealed class HoldoutSplitsTests : IDisposable
{
    private readonly bool _prevRandomizeEachRun;

    public HoldoutSplitsTests()
    {
        _prevRandomizeEachRun = SymbolicRegressionEngine.GpRandomizeBaseEachRun;
        SymbolicRegressionEngine.GpRandomizeBaseEachRun = false;
    }

    public void Dispose()
    {
        SymbolicRegressionEngine.GpRandomizeBaseEachRun = _prevRandomizeEachRun;
    }

    [Fact]
    public void RepeatedHoldout_Partition_DisjointAndSizes()
    {
        int n = 20;
        var mv = new MultiVarValidationConfig
        {
            Repeats = 4,
            TestFraction = 0.2,
            SplitMode = "holdout",
            ShuffleSeed = 999,
        };
        var sch = HoldoutSplits.CreateValidationSchedule(mv, n);
        Assert.False(sch.KFoldMode);
        Assert.Equal(4, sch.NumRuns);
        Assert.Equal(Math.Max(1, (int)Math.Round(n * 0.2)), sch.TestCountHoldout);

        for (int rep = 0; rep < sch.NumRuns; rep++)
        {
            HoldoutSplits.GetTrainTestIndexLists(sch, n, rep, out int[] tr, out int[] te);
            Assert.Equal(n, tr.Length + te.Length);
            var trSet = tr.ToHashSet();
            var teSet = te.ToHashSet();
            Assert.Empty(trSet.Intersect(teSet));
            Assert.Equal(tr.Length, trSet.Count);
            Assert.Equal(te.Length, teSet.Count);
            Assert.Equal(Enumerable.Range(0, n).ToHashSet(), tr.Concat(te).ToHashSet());
            foreach (int i in tr) Assert.InRange(i, 0, n - 1);
            foreach (int i in te) Assert.InRange(i, 0, n - 1);
        }
    }

    [Fact]
    public void RepeatedHoldout_Deterministic_WithFixedGpFlags()
    {
        int n = 15;
        var mv = new MultiVarValidationConfig { Repeats = 2, TestFraction = 0.25, SplitMode = "holdout" };
        var sch = HoldoutSplits.CreateValidationSchedule(mv, n);

        HoldoutSplits.GetTrainTestIndexLists(sch, n, 0, out int[] t1a, out int[] e1a);
        HoldoutSplits.GetTrainTestIndexLists(sch, n, 0, out int[] t1b, out int[] e1b);
        Assert.Equal(t1a, t1b);
        Assert.Equal(e1a, e1b);
    }

    [Fact]
    public void KFold_EachIndexInTestExactlyOnce()
    {
        int n = 100;
        int k = 5;
        var mv = new MultiVarValidationConfig
        {
            SplitMode = "kfold",
            KFolds = k,
            ShuffleSeed = 42,
        };
        var sch = HoldoutSplits.CreateValidationSchedule(mv, n);
        Assert.True(sch.KFoldMode);
        Assert.Equal(k, sch.NumRuns);
        Assert.NotNull(sch.FoldLen);
        var foldLen = sch.FoldLen;
        Assert.Equal(n, foldLen.Sum());

        var testCountPerIndex = new int[n];
        for (int rep = 0; rep < k; rep++)
        {
            HoldoutSplits.GetTrainTestIndexLists(sch, n, rep, out int[] tr, out int[] te);
            Assert.Equal(foldLen[rep], te.Length);
            Assert.Equal(n - te.Length, tr.Length);
            foreach (int i in te)
                testCountPerIndex[i]++;
        }

        Assert.All(testCountPerIndex, c => Assert.Equal(1, c));
    }

    [Fact]
    public void KFold_FoldSizes_DifferByAtMostOne()
    {
        int n = 103;
        var mv = new MultiVarValidationConfig { SplitMode = "kfold", KFolds = 10, ShuffleSeed = 1 };
        var sch = HoldoutSplits.CreateValidationSchedule(mv, n);
        int min = sch.FoldLen!.Min();
        int max = sch.FoldLen!.Max();
        Assert.InRange(max - min, 0, 1);
    }

    [Fact]
    public void MedianDoubles_Empty_IsNaN()
    {
        Assert.True(double.IsNaN(HoldoutSplits.MedianDoubles(new List<double>())));
    }

    [Fact]
    public void MedianDoubles_OddAndEven()
    {
        Assert.Equal(3.0, HoldoutSplits.MedianDoubles(new List<double> { 5, 1, 3 }));
        Assert.Equal(2.5, HoldoutSplits.MedianDoubles(new List<double> { 1, 2, 3, 4 }));
    }

    [Fact]
    public void CandidateKindTieRank_Ordering()
    {
        var names = new[] { "quadratic", "linear", "log-linear", "GP" };
        var ordered = names.OrderBy(HoldoutSplits.CandidateKindTieRank).ToArray();
        Assert.Equal(new[] { "linear", "quadratic", "log-linear", "GP" }, ordered);
    }
}
