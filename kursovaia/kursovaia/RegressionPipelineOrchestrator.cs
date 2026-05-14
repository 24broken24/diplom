using System;
using System.Collections.Generic;
using System.IO;
using ClosedXML.Excel;
using System.Linq;

namespace SymbolicRegression
{
    internal static class RegressionPipelineOrchestrator
    {
        /// <summary>Упорядочивает пары (x,y) по x (при равных x — по индексу) для эвристик паттерна и устойчивого hold-out.</summary>
        private static void SortOneVarByX(double[] x, double[] y)
        {
            int n = x.Length;
            if (n != y.Length) return;
            var order = Enumerable.Range(0, n).OrderBy(i => x[i]).ThenBy(i => i).ToArray();
            var xs = order.Select(i => x[i]).ToArray();
            var ys = order.Select(i => y[i]).ToArray();
            Array.Copy(xs, x, n);
            Array.Copy(ys, y, n);
        }

        /// <summary>
        /// Устойчивая сортировка строк матрицы X и вектора y по первому предиктору x₁ (при равных x₁ — по исходному индексу).
        /// Единый порядок строк для внешнего hold-out и отчёта (этап C плана).
        /// </summary>
        private static void SortMultivarRowsByFirstColumn(double[][] X, double[] y)
        {
            int n = X.Length;
            if (n == 0 || n != y.Length) return;
            var order = Enumerable.Range(0, n).OrderBy(i => X[i][0]).ThenBy(i => i).ToArray();
            var xNew = new double[n][];
            var yNew = new double[n];
            for (int r = 0; r < n; r++)
            {
                int src = order[r];
                xNew[r] = (double[])X[src].Clone();
                yNew[r] = y[src];
            }
            for (int r = 0; r < n; r++)
            {
                X[r] = xNew[r];
                y[r] = yNew[r];
            }
        }

        /// <summary>1-var GP: внешний hold-out по той же схеме, что multiVarValidation; иначе — полная выборка (малый n).</summary>
        internal static void ProcessOneVarRegression(
            string excelLabel,
            double[] xRaw,
            double[] yRaw,
            StreamWriter writer,
            AppRuntimeConfig runCfg,
            Dictionary<string, (double Rmse, double YRange)> benchmarkResults)
        {
            var xArr = (double[])xRaw.Clone();
            var yArr = (double[])yRaw.Clone();
            SortOneVarByX(xArr, yArr);
            int n = xArr.Length;

            writer.WriteLine("Данные X и Y (после сортировки по X):");
            for (int i = 0; i < Math.Min(10, n); i++)
                writer.WriteLine($"X: {xArr[i]:F4} | Y: {yArr[i]:F4}");
            if (n > 10)
                writer.WriteLine("...");

            double yRangeFull = Math.Max(yArr.Max() - yArr.Min(), 1e-12);
            bool useOneVarLogFit = runCfg.IncludeOneVarLogFit ?? true;
            writer.WriteLine($"Режим 1-var (gp_settings): logFit={(useOneVarLogFit ? "on" : "off")}");

            var probeForPattern = new SymbolicRegressionEngine(xArr, yArr);
            string detectedPattern = probeForPattern.GetDataPattern();
            writer.WriteLine($"Определенный тип данных: {detectedPattern}");

            var sch = HoldoutSplits.CreateValidationSchedule(runCfg.MultiVarValidation, n);
            int extTrainCount = n - sch.TestCountHoldout;
            bool useExternalHoldout = n >= 5 && extTrainCount >= 2 && sch.TestCountHoldout >= 1;

            if (!useExternalHoldout)
            {
                writer.WriteLine(
                    $"Внешний hold-out отключён (n={n}, внешний train={extTrainCount}): полная выборка; direct — конкурс gp-pattern vs gp-general по RMSE на всех точках; затем сравнение с log-fit при условиях ниже.");
                RunOneVarFullSampleNoHoldout(excelLabel, xArr, yArr, yRangeFull, useOneVarLogFit, writer, benchmarkResults);
                return;
            }

            writer.WriteLine();
            if (sch.KFoldMode)
                writer.WriteLine(
                    $"1-var внешняя валидация (как multiVarValidation): k-fold, k={sch.NumRuns}, shuffleSeed={runCfg.MultiVarValidation?.ShuffleSeed ?? 12345}");
            else
                writer.WriteLine(
                    $"1-var внешняя валидация (как multiVarValidation): повторяемый hold-out, repeats={sch.NumRuns}, testFraction={sch.TestFraction:F3} (~{sch.TestCountHoldout}/{n} в test)");
            writer.WriteLine(
                "  Внутри direct: два независимых GP на train — профиль по паттерну (AnalyzeDataPattern→grammar) и GrammarProfile.General; победитель сплита — по min RMSE_test (ничья → General).");
            writer.WriteLine(
                "  Выбор direct vs log-fit — по RMSE на внешнем test; итог по median RMSE_test; формула — с сплита с минимальным RMSE_test для выбранного типа.");
            writer.WriteLine(
                "  Внутри каждого Evolve() по-прежнему ~80/20 train/validation (отбор рестартов), только на внешнем train.");

            bool yAllPositiveFull = true;
            for (int i = 0; i < n; i++)
            {
                if (yArr[i] <= 0) { yAllPositiveFull = false; break; }
            }

            var directTestList = new List<double>();
            var patternTestList = new List<double>();
            var generalTestList = new List<double>();
            var logTestList = new List<double>();
            var directTrainList = new List<double>();
            var logTrainList = new List<double>();
            var directTrees = new ExpressionNode[sch.NumRuns];
            ExpressionNode?[] logTrees = new ExpressionNode[sch.NumRuns];
            var directFitness = new double[sch.NumRuns];
            var logFitness = new double[sch.NumRuns];

            for (int rep = 0; rep < sch.NumRuns; rep++)
            {
                HoldoutSplits.GetTrainTestIndexLists(sch, n, rep, out int[] trI, out int[] teI);
                int trN = trI.Length;
                int teN = teI.Length;
                var xTr = new double[trN];
                var yTr = new double[trN];
                for (int i = 0; i < trN; i++)
                {
                    int ix = trI[i];
                    xTr[i] = xArr[ix];
                    yTr[i] = yArr[ix];
                }
                var xTe = new double[teN];
                var yTe = new double[teN];
                for (int i = 0; i < teN; i++)
                {
                    int ix = teI[i];
                    xTe[i] = xArr[ix];
                    yTe[i] = yArr[ix];
                }

                SymbolicRegressionEngine.EvolutionTraceRunLabel = $"{excelLabel} [1-var gp-pattern rep{rep + 1}]";
                var engP = new SymbolicRegressionEngine(xTr, yTr);
                var (treeP, fitP) = engP.Evolve();
                double pTr = Program.ComputeRmseSingleVar(treeP, xTr, yTr);
                double pTe = Program.ComputeRmseSingleVar(treeP, xTe, yTe);
                patternTestList.Add(pTe);

                SymbolicRegressionEngine.EvolutionTraceRunLabel = $"{excelLabel} [1-var gp-general rep{rep + 1}]";
                var engG = new SymbolicRegressionEngine(xTr, yTr, SymbolicRegressionEngine.GrammarProfile.General);
                var (treeG, fitG) = engG.Evolve();
                double gTr = Program.ComputeRmseSingleVar(treeG, xTr, yTr);
                double gTe = Program.ComputeRmseSingleVar(treeG, xTe, yTe);
                generalTestList.Add(gTe);

                const double tieEps = 1e-12;
                bool takePattern = pTe + tieEps < gTe;
                ExpressionNode treeD;
                double fitD;
                double dTr;
                double dTe;
                if (takePattern)
                {
                    treeD = treeP;
                    fitD = fitP;
                    dTr = pTr;
                    dTe = pTe;
                }
                else
                {
                    treeD = treeG;
                    fitD = fitG;
                    dTr = gTr;
                    dTe = gTe;
                }

                directTrees[rep] = treeD;
                directFitness[rep] = fitD;
                directTrainList.Add(dTr);
                directTestList.Add(dTe);

                double lTr = double.NaN;
                double lTe = double.NaN;
                bool canLog = useOneVarLogFit && yAllPositiveFull && yTr.Length >= 5;
                if (canLog)
                {
                    for (int i = 0; i < trN; i++)
                    {
                        if (yTr[i] <= 0) { canLog = false; break; }
                    }
                }
                if (canLog)
                {
                    var logYTr = new double[trN];
                    for (int i = 0; i < trN; i++) logYTr[i] = Math.Log(yTr[i]);
                    SymbolicRegressionEngine.EvolutionTraceRunLabel = $"{excelLabel} [1-var log-fit rep{rep + 1}]";
                    var logEng = new SymbolicRegressionEngine(
                        xTr, logYTr, SymbolicRegressionEngine.GrammarProfile.General);
                    var (gTree, gFit) = logEng.Evolve();
                    var expW = new ExpressionNode
                    {
                        Operation = "exp",
                        Children = { gTree }
                    };
                    logTrees[rep] = expW;
                    logFitness[rep] = gFit;
                    lTr = Program.ComputeRmseSingleVar(expW, xTr, yTr);
                    lTe = Program.ComputeRmseSingleVar(expW, xTe, yTe);
                    logTrainList.Add(lTr);
                    logTestList.Add(lTe);
                }
                else
                {
                    logTrainList.Add(double.NaN);
                    logTestList.Add(double.NaN);
                    logTrees[rep] = null;
                    logFitness[rep] = double.NaN;
                }

                if (sch.NumRuns <= 5)
                {
                    writer.WriteLine($"\n--- 1-var сплит #{rep + 1}/{sch.NumRuns} ---");
                    writer.WriteLine($"  [gp-pattern] RMSE_train={pTr:F6}  RMSE_test={pTe:F6}");
                    writer.WriteLine($"  [gp-general] RMSE_train={gTr:F6}  RMSE_test={gTe:F6}");
                    writer.WriteLine(
                        $"  [direct best] RMSE_train={dTr:F6}  RMSE_test={dTe:F6}  ({(takePattern ? "pattern" : "general")})");
                    if (canLog && logTrees[rep] != null)
                        writer.WriteLine($"  [log-fit] RMSE_train={lTr:F6}  RMSE_test={lTe:F6}");
                    else if (useOneVarLogFit && yAllPositiveFull)
                        writer.WriteLine($"  [log-fit] пропуск (нет условий на train или слишком мало точек)");
                }
            }

            var validDirectTest = directTestList.Where(v => !double.IsNaN(v) && !double.IsInfinity(v)).ToList();
            double medDirectTest = HoldoutSplits.MedianDoubles(validDirectTest);
            var validPatternTest = patternTestList.Where(v => !double.IsNaN(v) && !double.IsInfinity(v)).ToList();
            double medPatternTest = validPatternTest.Count > 0 ? HoldoutSplits.MedianDoubles(validPatternTest) : double.NaN;
            var validGeneralTest = generalTestList.Where(v => !double.IsNaN(v) && !double.IsInfinity(v)).ToList();
            double medGeneralTest = validGeneralTest.Count > 0 ? HoldoutSplits.MedianDoubles(validGeneralTest) : double.NaN;
            var validLogTest = logTestList.Where(v => !double.IsNaN(v) && !double.IsInfinity(v)).ToList();
            double medLogTest = validLogTest.Count > 0 ? HoldoutSplits.MedianDoubles(validLogTest) : double.PositiveInfinity;

            string bestName = useOneVarLogFit && yAllPositiveFull && medLogTest < medDirectTest * 0.95
                ? "log-fit"
                : "direct";

            int bestRepIdx = -1;
            double bestScore = double.PositiveInfinity;
            for (int rep = 0; rep < sch.NumRuns; rep++)
            {
                if (bestName == "log-fit")
                {
                    if (logTrees[rep] == null) continue;
                    double te = logTestList[rep];
                    if (double.IsNaN(te) || double.IsInfinity(te)) continue;
                    if (te < bestScore)
                    {
                        bestScore = te;
                        bestRepIdx = rep;
                    }
                }
                else
                {
                    double te = directTestList[rep];
                    if (double.IsNaN(te) || double.IsInfinity(te)) continue;
                    if (te < bestScore)
                    {
                        bestScore = te;
                        bestRepIdx = rep;
                    }
                }
            }
            if (bestRepIdx < 0)
                bestRepIdx = 0;

            if (bestName == "log-fit" && logTrees[bestRepIdx] == null)
            {
                for (int rep = 0; rep < sch.NumRuns; rep++)
                {
                    if (logTrees[rep] == null) continue;
                    double te = logTestList[rep];
                    if (double.IsNaN(te) || double.IsInfinity(te)) continue;
                    bestRepIdx = rep;
                    break;
                }
            }

            if (bestName == "log-fit" && logTrees[bestRepIdx] == null)
            {
                bestName = "direct";
                bestScore = double.PositiveInfinity;
                bestRepIdx = -1;
                for (int rep = 0; rep < sch.NumRuns; rep++)
                {
                    double te = directTestList[rep];
                    if (double.IsNaN(te) || double.IsInfinity(te)) continue;
                    if (te < bestScore) { bestScore = te; bestRepIdx = rep; }
                }
                if (bestRepIdx < 0) bestRepIdx = 0;
            }

            ExpressionNode finalTree = bestName == "log-fit" && logTrees[bestRepIdx] != null
                ? logTrees[bestRepIdx]!
                : directTrees[bestRepIdx];
            double finalFitness = bestName == "log-fit" && logTrees[bestRepIdx] != null
                ? logFitness[bestRepIdx]
                : directFitness[bestRepIdx];

            HoldoutSplits.GetTrainTestIndexLists(sch, n, bestRepIdx, out int[] brTr, out int[] brTe);
            int brTrN = brTr.Length;
            int brTeN = brTe.Length;
            var xBrTr = new double[brTrN];
            var yBrTr = new double[brTrN];
            for (int i = 0; i < brTrN; i++)
            {
                int ix = brTr[i];
                xBrTr[i] = xArr[ix];
                yBrTr[i] = yArr[ix];
            }
            var xBrTe = new double[brTeN];
            var yBrTe = new double[brTeN];
            for (int i = 0; i < brTeN; i++)
            {
                int ix = brTe[i];
                xBrTe[i] = xArr[ix];
                yBrTe[i] = yArr[ix];
            }

            double rmseTrainRep = Program.ComputeRmseSingleVar(finalTree, xBrTr, yBrTr);
            double rmseTestRep = Program.ComputeRmseSingleVar(finalTree, xBrTe, yBrTe);
            double medianWinnerTest = bestName == "log-fit" && validLogTest.Count > 0 ? medLogTest : medDirectTest;

            writer.WriteLine();
            writer.WriteLine(
                $"Сводка 1-var: median RMSE_test [gp-pattern]={(validPatternTest.Count > 0 ? medPatternTest.ToString("F6") : "N/A")}; " +
                $"[gp-general]={(validGeneralTest.Count > 0 ? medGeneralTest.ToString("F6") : "N/A")}; " +
                $"[direct=min per split]={medDirectTest:F6}; median RMSE_test [log-fit]={(validLogTest.Count > 0 ? medLogTest.ToString("F6") : "N/A")}");
            writer.WriteLine(
                $"Победитель по median RMSE_test: [{bestName}] (формула со сплита #{bestRepIdx + 1}, минимальный RMSE_test среди сплитов для этого типа)");
            if (bestName == "log-fit" && logTrees[bestRepIdx] != null && logTrees[bestRepIdx]!.Children.Count > 0)
            {
                var g = logTrees[bestRepIdx]!.Children[0];
                var prettyG = Program.ExpandAndSimplifyBaseFormula(g, null);
                writer.WriteLine($"  log-fit развёртка: exp({prettyG})");
            }
            var prettyTree = Program.ExpandAndSimplifyBaseFormula(finalTree, null);
            writer.WriteLine($"\nНайденная формула [{bestName}]: {prettyTree}");
            writer.WriteLine($"{BenchmarkReporting.GpTrainObjectiveReportLabel}: {finalFitness:F4}");
            writer.WriteLine($"RMSE (train), сплит #{bestRepIdx + 1}: {rmseTrainRep:F6}");
            writer.WriteLine($"RMSE (test), сплит #{bestRepIdx + 1}: {rmseTestRep:F6}");
            writer.WriteLine($"Median RMSE (test) по всем сплитам для [{bestName}]: {medianWinnerTest:F6}  (для benchmark summary)");
            writer.WriteLine($"RMSE / yRange (полный ряд, оценка через median test): {medianWinnerTest / yRangeFull:E3}");

            if (!double.IsNaN(medianWinnerTest) && !double.IsInfinity(medianWinnerTest))
                benchmarkResults[excelLabel] = (medianWinnerTest, yRangeFull);
        }

        /// <summary>Полный ряд без внешнего hold-out (малый n): выбор модели по RMSE на всех точках (как в старой версии).</summary>
        private static void RunOneVarFullSampleNoHoldout(
            string excelLabel,
            double[] xArr,
            double[] yArr,
            double yRange,
            bool useOneVarLogFit,
            StreamWriter writer,
            Dictionary<string, (double Rmse, double YRange)> benchmarkResults)
        {
            SymbolicRegressionEngine.EvolutionTraceRunLabel = $"{excelLabel} [1-var gp-pattern]";
            var (treeP, fitP) = new SymbolicRegressionEngine(xArr, yArr).Evolve();
            double rmseP = Program.ComputeRmseSingleVar(treeP, xArr, yArr);

            SymbolicRegressionEngine.EvolutionTraceRunLabel = $"{excelLabel} [1-var gp-general]";
            var (treeG, fitG) = new SymbolicRegressionEngine(xArr, yArr, SymbolicRegressionEngine.GrammarProfile.General).Evolve();
            double rmseG = Program.ComputeRmseSingleVar(treeG, xArr, yArr);

            const double tieEps = 1e-12;
            bool takePattern = rmseP + tieEps < rmseG;
            ExpressionNode directTree = takePattern ? treeP : treeG;
            double directFitness = takePattern ? fitP : fitG;
            double directRmse = takePattern ? rmseP : rmseG;

            writer.WriteLine(
                $"Конкурс GP (полная выборка): pattern RMSE={rmseP:F6}, general RMSE={rmseG:F6}; direct — {(takePattern ? "pattern" : "general")}.");

            ExpressionNode bestTree = directTree;
            double bestFitness = directFitness;
            double bestRmse = directRmse;
            string method = "direct";

            bool yAllPositive = yArr.All(y => y > 0);

            if (useOneVarLogFit && yAllPositive && yArr.Length >= 5)
            {
                var logY = new double[yArr.Length];
                for (int i = 0; i < yArr.Length; i++) logY[i] = Math.Log(yArr[i]);

                SymbolicRegressionEngine.EvolutionTraceRunLabel = $"{excelLabel} [1-var log-fit]";
                var logEngine = new SymbolicRegressionEngine(
                    xArr, logY, SymbolicRegressionEngine.GrammarProfile.General);
                var (gTree, gFitness) = logEngine.Evolve();

                var expWrapped = new ExpressionNode
                {
                    Operation = "exp",
                    Children = { gTree }
                };

                double logRmse = Program.ComputeRmseSingleVar(expWrapped, xArr, yArr);
                var prettyG = Program.ExpandAndSimplifyBaseFormula(gTree, null);
                writer.WriteLine($"  log-fit кандидат: y = exp({prettyG}), RMSE_full={logRmse:F6}");

                if (!double.IsNaN(logRmse) && !double.IsInfinity(logRmse) && logRmse < bestRmse * 0.95)
                {
                    bestTree = expWrapped;
                    bestFitness = gFitness;
                    bestRmse = logRmse;
                    method = "log-fit";
                    writer.WriteLine("  log-преобразование улучшило фит на полной выборке, переключаемся на exp(g(x))");
                }
            }

            var prettyTree = Program.ExpandAndSimplifyBaseFormula(bestTree, null);
            writer.WriteLine($"\nНайденная формула [{method}]: {prettyTree}");
            writer.WriteLine($"{BenchmarkReporting.GpTrainObjectiveReportLabel}: {bestFitness:F4}");
            writer.WriteLine($"RMSE (полная выборка): {bestRmse:F6}");
            writer.WriteLine($"RMSE / yRange: {bestRmse / yRange:E3}");
            if (!double.IsNaN(bestRmse) && !double.IsInfinity(bestRmse))
                benchmarkResults[excelLabel] = (bestRmse, yRange);
        }




        // Обработка множественной регрессии
        /// <param name="traceWorkbookLabel">Имя .xlsx для префикса gp_evolution_trace (фаза B); null — взять текущий EvolutionTraceRunLabel или «multi-var».</param>
        internal static (double Rmse, double YRange)? ProcessMultipleRegression(IXLWorksheet worksheet, StreamWriter writer, AppRuntimeConfig runCfg, string? traceWorkbookLabel = null)
        {
            try
            {
                int colCount = worksheet.LastColumnUsed().ColumnNumber();
                int rowCount = worksheet.LastRowUsed().RowNumber();
                
                writer.WriteLine($"Количество переменных: {colCount - 1}");
                writer.WriteLine($"Количество наблюдений: {rowCount - 1}");
                
                // Создаем матрицу данных
                var data = new List<double[]>();
                int startRow = 2; // Пропускаем заголовок
                
                for (int row = startRow; row <= rowCount; row++)
                {
                    var rowData = new double[colCount];
                    bool validRow = true;
                    
                    for (int col = 1; col <= colCount; col++)
                    {
                        if (double.TryParse(worksheet.Cell(row, col).GetString(), out double value))
                        {
                            rowData[col - 1] = value;
                        }
                        else
                        {
                            validRow = false;
                            break;
                        }
                    }
                    
                    if (validRow)
                    {
                        data.Add(rowData);
                    }
                }
                
                if (data.Count < 2)
                {
                    writer.WriteLine("Недостаточно данных для множественной регрессии!");
                    return null;
                }
                
                // Для multi-var (2+ предикторов) превью строк — после сортировки по x₁ (см. ниже).
                if (colCount < 3)
                {
                    writer.WriteLine("Первые 10 строк данных:");
                    for (int i = 0; i < Math.Min(10, data.Count); i++)
                    {
                        var row = data[i];
                        writer.Write($"Строка {i + 1}: ");
                        for (int j = 0; j < row.Length; j++)
                            writer.Write($"{row[j]:F4} ");
                        writer.WriteLine();
                    }
                }

                // Общий (не подогнанный) случай:
                // - если 2..N предикторов и Y: символьная регрессия по n переменным (с режимом суперпозиции при наличии base_functions.txt)
                // - множественная линейная регрессия остается baseline ниже
                if (colCount >= 3)
                {
                    int n = data.Count;
                    int p = colCount - 1;
                    var X = new double[n][];
                    var yArr = new double[n];
                    for (int i = 0; i < n; i++)
                    {
                        var rowX = new double[p];
                        for (int j = 0; j < p; j++)
                            rowX[j] = data[i][j];
                        X[i] = rowX;
                        yArr[i] = data[i][p];
                    }

                    SortMultivarRowsByFirstColumn(X, yArr);
                    writer.WriteLine();
                    writer.WriteLine("Первые 10 строк после сортировки по x₁:");
                    for (int i = 0; i < Math.Min(10, n); i++)
                    {
                        writer.Write($"Строка {i + 1}: ");
                        for (int j = 0; j < p; j++)
                            writer.Write($"{X[i][j]:F4} ");
                        writer.Write($"{yArr[i]:F4}");
                        writer.WriteLine();
                    }
                    if (n > 10)
                        writer.WriteLine("...");

                    bool useClassicBaselines = runCfg.IncludeClassicRegressionBaselines ?? true;
                    var sch = HoldoutSplits.CreateValidationSchedule(runCfg.MultiVarValidation, n);

                    writer.WriteLine();
                    if (sch.KFoldMode)
                        writer.WriteLine(
                            $"Режим сравнения (gp_settings): k-fold, k={sch.NumRuns}, shuffleSeed={runCfg.MultiVarValidation?.ShuffleSeed ?? 12345}");
                    else
                        writer.WriteLine(
                            $"Режим сравнения (gp_settings): повторяемый hold-out, repeats={sch.NumRuns}, testFraction={sch.TestFraction:F3} (~{sch.TestCountHoldout}/{n} в test)");
                    writer.WriteLine(
                        $"  classicRegressionBaselines: {(useClassicBaselines ? "on" : "off")}");

                    writer.WriteLine();
                    var basePath = Path.Combine(Environment.CurrentDirectory, "base_functions.txt");
                    var baseLoadWarnings = new List<string>();
                    var baseFunctionsProbe =
                        SymbolicRegressionEngine.LoadBaseFunctions(basePath, p, baseLoadWarnings);

                    writer.WriteLine("Суперпозиция (диагностика загрузки base_functions.txt):");
                    writer.WriteLine($"  Путь: {basePath}");
                    writer.WriteLine($"  Файл найден: {File.Exists(basePath)}");
                    writer.WriteLine($"  Загружено баз при p={p}: {baseFunctionsProbe.Count}");
                    if (baseLoadWarnings.Count > 0)
                    {
                        writer.WriteLine("  Предупреждения при разборе строк:");
                        foreach (var w in baseLoadWarnings)
                            writer.WriteLine($"    - {w}");
                    }
                    else
                        writer.WriteLine("  Предупреждения при разборе строк: нет.");

                    var candidatesPerRep =
                        new List<
                            List<(string Name, ExpressionNode Tree, List<ExpressionNode>? Bases, double TrainRmse,
                                double TestRmse, double GpFitness)>>();

                    for (int rep = 0; rep < sch.NumRuns; rep++)
                    {
                        HoldoutSplits.GetTrainTestIndexLists(sch, n, rep, out int[] trainIdx, out int[] testIdx);
                        int trainCount = trainIdx.Length;
                        int testCountArr = testIdx.Length;

                        var XTrain = new double[trainCount][];
                        var yTrain = new double[trainCount];
                        for (int i = 0; i < trainCount; i++)
                        {
                            int idx = trainIdx[i];
                            XTrain[i] = X[idx];
                            yTrain[i] = yArr[idx];
                        }

                        var XTest = new double[testCountArr][];
                        var yTest = new double[testCountArr];
                        for (int i = 0; i < testCountArr; i++)
                        {
                            int idx = testIdx[i];
                            XTest[i] = X[idx];
                            yTest[i] = yArr[idx];
                        }

                        var baseFunctions = SymbolicRegressionEngine.LoadBaseFunctions(basePath, p, null);
                        var symbolicRegression = new SymbolicRegressionEngine(XTrain, yTrain, baseFunctions, superposition: true);
                        if (rep == 0)
                            writer.WriteLine($"\nОпределенный тип данных: {symbolicRegression.GetDataPattern()}");

                        string wbTag = !string.IsNullOrWhiteSpace(traceWorkbookLabel)
                            ? traceWorkbookLabel.Trim()
                            : (string.IsNullOrWhiteSpace(SymbolicRegressionEngine.EvolutionTraceRunLabel)
                                ? "multi-var"
                                : SymbolicRegressionEngine.EvolutionTraceRunLabel.Trim());
                        string splitModeTag = sch.KFoldMode ? "kfold" : "holdout";
                        SymbolicRegressionEngine.EvolutionTraceRunLabel =
                            $"{wbTag} [multi-var superposition {splitModeTag} rep{rep + 1}/{sch.NumRuns}]";

                        var (gpTree, gpFitness) = symbolicRegression.Evolve();

                        var candidates =
                            new List<(string Name, ExpressionNode Tree, List<ExpressionNode>? Bases, double TrainRmse,
                                double TestRmse, double GpFitness)>
                            {
                                ("GP", gpTree, baseFunctions,
                                    Program.ComputeRmseN(gpTree, XTrain, yTrain, baseFunctions),
                                    Program.ComputeRmseN(gpTree, XTest, yTest, baseFunctions), gpFitness)
                            };

                        if (useClassicBaselines)
                        {
                            var linTree = Program.BuildLinearBaseline(XTrain, yTrain, p);
                            if (linTree != null)
                                candidates.Add(("linear", linTree, null,
                                    Program.ComputeRmseN(linTree, XTrain, yTrain, null),
                                    Program.ComputeRmseN(linTree, XTest, yTest, null), double.NaN));

                            var quadTree = Program.BuildQuadraticBaseline(XTrain, yTrain, p);
                            if (quadTree != null)
                                candidates.Add(("quadratic", quadTree, null,
                                    Program.ComputeRmseN(quadTree, XTrain, yTrain, null),
                                    Program.ComputeRmseN(quadTree, XTest, yTest, null), double.NaN));

                            var logTree = Program.BuildLogLinearBaseline(XTrain, yTrain, p);
                            if (logTree != null)
                                candidates.Add(("log-linear", logTree, null,
                                    Program.ComputeRmseN(logTree, XTrain, yTrain, null),
                                    Program.ComputeRmseN(logTree, XTest, yTest, null), double.NaN));
                        }

                        candidatesPerRep.Add(candidates);

                        if (sch.NumRuns <= 5)
                        {
                            writer.WriteLine($"\n--- Сплит/фолд #{rep + 1}/{sch.NumRuns} ---");
                            foreach (var c in candidates)
                            {
                                var prettyC = Program.ExpandAndSimplifyBaseFormula(c.Tree, c.Bases);
                                string gpBaseHint = "";
                                if (c.Name.Equals("GP", StringComparison.OrdinalIgnoreCase))
                                {
                                    if (baseFunctionsProbe.Count == 0)
                                        gpBaseHint = "  [базы: не загружены — терминалов base нет]";
                                    else if (Program.ExpressionTreeUsesBaseCalls(c.Tree))
                                        gpBaseHint = "  [базы: в дереве есть узлы f1..fk]";
                                    else
                                        gpBaseHint = "  [базы: загружены, дерево GP без узлов base]";
                                }

                                writer.WriteLine(
                                    $"  [{c.Name}] {prettyC}  RMSE_train={c.TrainRmse:F6}  RMSE_test={c.TestRmse:F6}{gpBaseHint}");
                            }
                        }
                    }

                    writer.WriteLine();
                    writer.WriteLine("Сводка: использование баз в финальном дереве GP по сплитам:");
                    if (baseFunctionsProbe.Count == 0)
                    {
                        writer.WriteLine(
                            "  Базы не загружены — суперпозиция по определению выключена; узлы base в GP отсутствуют.");
                    }
                    else
                    {
                        int withBaseNodes = 0;
                        for (int ri = 0; ri < candidatesPerRep.Count; ri++)
                        {
                            var gpPair = candidatesPerRep[ri]
                                .FirstOrDefault(c => c.Name.Equals("GP", StringComparison.OrdinalIgnoreCase));
                            if (gpPair.Tree != null && Program.ExpressionTreeUsesBaseCalls(gpPair.Tree))
                                withBaseNodes++;
                        }

                        writer.WriteLine(
                            $"  Всего сплитов: {candidatesPerRep.Count}; дерево GP содержит узлы base: {withBaseNodes} из {candidatesPerRep.Count}.");
                    }

                    var nameSet = new HashSet<string>(StringComparer.Ordinal);
                    foreach (var rep in candidatesPerRep)
                    foreach (var c in rep)
                        nameSet.Add(c.Name);

                    var medianTest = new Dictionary<string, double>(StringComparer.Ordinal);
                    writer.WriteLine("\nСводка по валидации (median RMSE_test по сплитам; IQR по RMSE_test):");
                    foreach (var nm in nameSet.OrderBy(s => s, StringComparer.Ordinal))
                    {
                        var testVals = candidatesPerRep
                            .SelectMany(r => r.Where(c => c.Name == nm).Select(c => c.TestRmse))
                            .Where(v => !double.IsNaN(v) && !double.IsInfinity(v))
                            .ToList();
                        if (testVals.Count == 0)
                            continue;
                        medianTest[nm] = HoldoutSplits.MedianDoubles(testVals);
                        double iqrT = HoldoutSplits.IqrDoubles(testVals);
                        var trainVals = candidatesPerRep
                            .SelectMany(r => r.Where(c => c.Name == nm).Select(c => c.TrainRmse))
                            .Where(v => !double.IsNaN(v) && !double.IsInfinity(v))
                            .ToList();
                        double medTr = trainVals.Count > 0 ? HoldoutSplits.MedianDoubles(trainVals) : double.NaN;
                        writer.WriteLine($"  [{nm}] median RMSE_train={medTr:F6}  median RMSE_test={medianTest[nm]:F6}  IQR_test={iqrT:F6}");
                    }

                    if (medianTest.Count == 0)
                    {
                        writer.WriteLine("Нет валидных RMSE_test по кандидатам.");
                        return null;
                    }

                    string bestName = medianTest
                        .OrderBy(kv => kv.Value)
                        .ThenBy(kv => HoldoutSplits.CandidateKindTieRank(kv.Key))
                        .First().Key;

                    int bestRepIdx = -1;
                    double bestRepTest = double.PositiveInfinity;
                    double gpFitnessForRep = double.NaN;
                    ExpressionNode bestTree = null!;
                    List<ExpressionNode>? bestBases = null;

                    for (int rep = 0; rep < candidatesPerRep.Count; rep++)
                    {
                        foreach (var c in candidatesPerRep[rep])
                        {
                            if (c.Name != bestName)
                                continue;
                            if (double.IsNaN(c.TestRmse) || double.IsInfinity(c.TestRmse))
                                continue;
                            if (c.TestRmse < bestRepTest)
                            {
                                bestRepTest = c.TestRmse;
                                bestRepIdx = rep;
                                bestTree = c.Tree;
                                bestBases = c.Bases;
                                gpFitnessForRep = c.GpFitness;
                            }
                        }
                    }

                    if (bestRepIdx < 0)
                    {
                        writer.WriteLine("Не удалось выбрать представителя модели для отчёта.");
                        return null;
                    }

                    HoldoutSplits.GetTrainTestIndexLists(sch, n, bestRepIdx, out int[] trI, out int[] teI);
                    int trN = trI.Length;
                    int teN = teI.Length;
                    var XTr = new double[trN][];
                    var yTr = new double[trN];
                    for (int i = 0; i < trN; i++)
                    {
                        int ix = trI[i];
                        XTr[i] = X[ix];
                        yTr[i] = yArr[ix];
                    }

                    var XTe = new double[teN][];
                    var yTe = new double[teN];
                    for (int i = 0; i < teN; i++)
                    {
                        int ix = teI[i];
                        XTe[i] = X[ix];
                        yTe[i] = yArr[ix];
                    }

                    var expandedTree = Program.ExpandAndSimplifyBaseFormula(bestTree, bestBases);
                    writer.WriteLine($"\nПобедитель по median RMSE_test: [{bestName}] (формула — с сплита #{bestRepIdx + 1}, минимальный RMSE_test среди сплитов для этого типа)");
                    writer.WriteLine($"Найденная формула [{bestName}]: {expandedTree}");
                    if (p == 2 && Program.NearlyCollinearPredictors(X, p, out string? colNote) && colNote != null)
                        writer.WriteLine(colNote);
                    if (bestName.Equals("GP", StringComparison.OrdinalIgnoreCase))
                        writer.WriteLine($"{BenchmarkReporting.GpTrainObjectiveReportLabel}: {gpFitnessForRep:F4}");
                    else
                    {
                        double trRmseRep = Program.ComputeRmseN(bestTree, XTr, yTr, bestBases);
                        double mseTrainWinner = trRmseRep * trRmseRep;
                        writer.WriteLine(
                            $"Выбран кандидат [{bestName}] . MSE на обучении этой модели на выбранном сплите: {mseTrainWinner:F6} (RMSE_train={trRmseRep:F6})");
                    }

                    double sseTrain = 0;
                    double meanYTrain = yTr.Average();
                    double sstTrain = 0;
                    int validTrain = 0;
                    for (int i = 0; i < trN; i++)
                    {
                        double predicted = Program.EvaluateTreeN(bestTree, XTr[i], bestBases);
                        if (double.IsNaN(predicted) || double.IsInfinity(predicted))
                            continue;
                        double diff = yTr[i] - predicted;
                        sseTrain += diff * diff;

                        double dy = yTr[i] - meanYTrain;
                        sstTrain += dy * dy;
                        validTrain++;
                    }

                    double sseTest = 0;
                    int validTest = 0;
                    for (int i = 0; i < teN; i++)
                    {
                        double predicted = Program.EvaluateTreeN(bestTree, XTe[i], bestBases);
                        if (double.IsNaN(predicted) || double.IsInfinity(predicted))
                            continue;
                        double diff = yTe[i] - predicted;
                        sseTest += diff * diff;
                        validTest++;
                    }

                    if (validTrain == 0 || validTest == 0)
                    {
                        writer.WriteLine("RMSE: не число (нет валидных предсказаний)");
                        writer.WriteLine("R² = не число");
                        return null;
                    }

                    double rmseTrainRep = Math.Sqrt(sseTrain / validTrain);
                    double r2Train = sstTrain > 0 ? 1 - (sseTrain / sstTrain) : 1.0;
                    double rmseTestRep = Math.Sqrt(sseTest / validTest);
                    double overfitGap = rmseTestRep - rmseTrainRep;
                    double overfitWarnThreshold = Math.Max(0.05, rmseTrainRep * 0.5);
                    string overfitStatus = overfitGap > overfitWarnThreshold ? "WARNING" : "OK";

                    double yRangeMulti = Math.Max(yArr.Max() - yArr.Min(), 1e-12);
                    double relRmseTestRep = rmseTestRep / yRangeMulti;
                    double medianWinnerTest = medianTest[bestName];

                    writer.WriteLine($"RMSE (train), сплит #{bestRepIdx + 1}: {rmseTrainRep:F6}");
                    writer.WriteLine($"R² (train) = {r2Train:F4}");
                    writer.WriteLine($"RMSE (test), сплит #{bestRepIdx + 1}: {rmseTestRep:F6}");
                    writer.WriteLine($"Median RMSE (test) по всем сплитам для [{bestName}]: {medianWinnerTest:F6}  (для benchmark summary)");
                    writer.WriteLine($"RMSE / yRange (test, этот сплит): {relRmseTestRep:E3}");
                    writer.WriteLine($"overfit_gap = {overfitGap:F6}");
                    writer.WriteLine($"overfit_status = {overfitStatus}");
                    return (medianWinnerTest, yRangeMulti);
                }

                var coefficients = Program.CalculateMultipleRegressionCoefficients(data);

                writer.WriteLine("\nКоэффициенты множественной линейной регрессии:");
                writer.Write("Y = ");
                for (int i = 0; i < coefficients.Length; i++)
                {
                    if (i == 0)
                    {
                        writer.Write($"{coefficients[i]:F4}");
                    }
                    else
                    {
                        if (coefficients[i] >= 0)
                            writer.Write($" + {coefficients[i]:F4}*X{i}");
                        else
                            writer.Write($" - {Math.Abs(coefficients[i]):F4}*X{i}");
                    }
                }
                writer.WriteLine();

                double rSquared = Program.CalculateRSquared(data, coefficients);
                writer.WriteLine($"R² = {rSquared:F4}");
                double linRmse = Program.CalculateRmseForLinearModel(data, coefficients);
                int pCol = data[0].Length - 1;
                double yMin = double.PositiveInfinity, yMax = double.NegativeInfinity;
                foreach (var r in data)
                {
                    double v = r[pCol];
                    if (v < yMin) yMin = v;
                    if (v > yMax) yMax = v;
                }
                double yRangeLin = Math.Max(yMax - yMin, 1e-12);
                writer.WriteLine($"RMSE / yRange: {linRmse / yRangeLin:E3}");
                return (linRmse, yRangeLin);
            }
            catch (Exception ex)
            {
                writer.WriteLine($"Ошибка при обработке множественной регрессии: {ex.Message}");
                return null;
            }
        }
    }
}
