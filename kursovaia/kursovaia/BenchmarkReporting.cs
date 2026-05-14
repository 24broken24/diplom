using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;

namespace SymbolicRegression
{
    internal static class BenchmarkReporting
    {
        internal const string GpTrainObjectiveReportLabel =
            "Значение целевой функции GP на обучении (MSE с штрафами за сложность и долю невалидных предсказаний; не BIC/AIC)";

        /// Запас, если в gp_settings не задан evaluationMaxRelativeRmse
        internal const double DefaultEvaluationMaxRelativeRmse = 0.005;

        /// Запас, если в gp_settings не задан evaluationMaxAbsRmse
        internal const double DefaultEvaluationMaxAbsRmse = 1.0;

        /// Те же правила, что в сводке: PASS, если RMSE ≤ maxAbs или rel ≤ maxRel
        internal static (bool pass, string passReason, double relRmse) EvaluateBenchmarkRow(
            double rmse, double yRange, AppRuntimeConfig cfg)
        {
            double maxRel = cfg.EvaluationMaxRelativeRmse ?? DefaultEvaluationMaxRelativeRmse;
            double maxAbs = cfg.EvaluationMaxAbsRmse ?? DefaultEvaluationMaxAbsRmse;
            double rel = yRange > 0 && !double.IsNaN(yRange) && !double.IsInfinity(yRange)
                ? rmse / yRange
                : double.NaN;
            bool okAbs = !double.IsNaN(rmse) && !double.IsInfinity(rmse) && rmse <= maxAbs;
            bool okRel = !double.IsNaN(rel) && !double.IsInfinity(rel) && rel <= maxRel;
            bool ok = okAbs || okRel;
            string passReason = ok ? (okAbs && okRel ? "abs+rel" : (okAbs ? "abs" : "rel")) : "fail";
            return (ok, passReason, rel);
        }

        internal static void WriteBenchmarkSummary(StreamWriter writer, Dictionary<string, (double Rmse, double YRange)> benchmarkResults, AppRuntimeConfig cfg)
        {
            writer.WriteLine("=== Benchmark summary ===");
            double maxRel = cfg.EvaluationMaxRelativeRmse ?? DefaultEvaluationMaxRelativeRmse;
            double maxAbs = cfg.EvaluationMaxAbsRmse ?? DefaultEvaluationMaxAbsRmse;
            writer.WriteLine(
                "В таблице — все .xlsx, по которым в этом запуске получена итоговая RMSE. " +
                "Критерии PASS одинаковы для всех файлов (см. gp_settings.json → evaluationMaxAbsRmse, evaluationMaxRelativeRmse): " +
                $"RMSE ≤ maxAbs или RMSE/yRange ≤ maxRel; если ключ не задан, maxAbs / maxRel = " +
                $"{DefaultEvaluationMaxAbsRmse:F3} / {DefaultEvaluationMaxRelativeRmse:F3} (запас в коде).");
            int pass = 0;
            int totalProcessed = benchmarkResults.Count;

            foreach (var kv in benchmarkResults.OrderBy(x => x.Key, StringComparer.OrdinalIgnoreCase))
            {
                var entry = kv.Value;

                double rmse = entry.Rmse;
                var (ok, passReason, rel) = EvaluateBenchmarkRow(rmse, entry.YRange, cfg);
                if (ok) pass++;

                writer.WriteLine(
                    $"{kv.Key}: RMSE={FormatRmse(rmse)}, rel={rel:E3}, " +
                    $"maxAbs={maxAbs:F6}, maxRel={maxRel:F6}, " +
                    $"status={(ok ? "PASS" : "FAIL")} ({passReason})");
            }

            writer.WriteLine();
            if (totalProcessed == 0)
                writer.WriteLine("Benchmark pass rate: нет обработанных файлов с валидной RMSE.");
            else
                writer.WriteLine($"Benchmark pass rate (обработанные файлы, N={totalProcessed}): {pass}/{totalProcessed}");
        }

        // Большие RMSE показываем в экспоненциальном виде.
        private static string FormatRmse(double v)
        {
            if (double.IsNaN(v) || double.IsInfinity(v)) return v.ToString();
            double abs = Math.Abs(v);
            if (abs >= 1e6 || (abs > 0 && abs < 1e-4)) return v.ToString("E3");
            return v.ToString("F6");
        }
    }
}
