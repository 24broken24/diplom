using System;
using System.IO;
using ClosedXML.Excel;
using MathNet.Numerics;
using MathNet.Numerics.LinearRegression;
using MathNet.Numerics.Statistics;
using MathNet.Numerics.LinearAlgebra;
using System.Collections.Generic;
using System.Linq;
using System.Text;

namespace SymbolicRegression
{
    class Program
    {
        static void Main()
        {
            var runtimeConfig = RuntimeConfigLoader.Load();
            SymbolicRegressionEngine.SetExternalProfileOverrides(runtimeConfig.GpProfiles);
            SymbolicRegressionEngine.GpRandomBaseSeed = runtimeConfig.GpRandomBaseSeed ?? 12345;
            SymbolicRegressionEngine.GpRandomizeBaseEachRun = runtimeConfig.GpRandomizeBaseEachRun;
            SymbolicRegressionEngine.GlobalEvolutionTrace = runtimeConfig.GpEvolutionTrace;
            if (runtimeConfig.GpEvolutionTrace?.Enabled == true && !string.IsNullOrWhiteSpace(runtimeConfig.GpEvolutionTrace.LogFile))
            {
                string tracePath = Path.IsPathRooted(runtimeConfig.GpEvolutionTrace.LogFile)
                    ? runtimeConfig.GpEvolutionTrace.LogFile
                    : Path.Combine(Environment.CurrentDirectory, runtimeConfig.GpEvolutionTrace.LogFile);
                File.WriteAllText(tracePath,
                    $"# gp_evolution_trace — запуск {DateTime.Now:u} (топ деревьев по fitness за поколение)\n\n",
                    Encoding.UTF8);
            }

            // автоматически читаем все .xlsx из текущей папки запуска
            var excelFiles = Directory.GetFiles(Environment.CurrentDirectory, "*.xlsx")
                .Where(p => !Path.GetFileName(p).StartsWith("~$"))
                .OrderBy(p => Path.GetFileName(p), StringComparer.OrdinalIgnoreCase)
                .ToArray();
            string outputPath = "result.txt";

            using (var writer = new StreamWriter(outputPath, false))
            {
                var benchmarkResults = new Dictionary<string, (double Rmse, double YRange)>(StringComparer.OrdinalIgnoreCase);
                if (excelFiles.Length == 0)
                {
                    writer.WriteLine("В текущей папке нет .xlsx файлов для обработки.");
                }

                foreach (var excelPath in excelFiles)
                {
                    writer.WriteLine($"Файл: {Path.GetFileName(excelPath)}");
                    try
                    {
                        var workbook = new XLWorkbook(excelPath);
                        var worksheet = workbook.Worksheet(1);

                        int colCount = worksheet.LastColumnUsed().ColumnNumber();
                        
                        if (colCount > 2)
                        {
                            var multiResult = RegressionPipelineOrchestrator.ProcessMultipleRegression(
                                worksheet, writer, runtimeConfig, Path.GetFileName(excelPath));
                            if (multiResult.HasValue && !double.IsNaN(multiResult.Value.Rmse) && !double.IsInfinity(multiResult.Value.Rmse))
                                benchmarkResults[Path.GetFileName(excelPath)] = multiResult.Value;
                        writer.WriteLine();
                        continue;
                    }

                        var xList = new List<double>();
                        var yList = new List<double>();

                    int row = 1;
                    bool hasHeader = !double.TryParse(worksheet.Cell(row, 1).GetString(), out _)
                                      || !double.TryParse(worksheet.Cell(row, 2).GetString(), out _);
                    if (hasHeader) row++;

                    while (!worksheet.Cell(row, 1).IsEmpty() && !worksheet.Cell(row, 2).IsEmpty())
                    {
                        if (double.TryParse(worksheet.Cell(row, 1).GetString(), out double xVal) &&
                            double.TryParse(worksheet.Cell(row, 2).GetString(), out double yVal))
                        {
                            xList.Add(xVal);
                            yList.Add(yVal);
                        }
                        row++;
                    }

                        double[] xArr = xList.ToArray();
                        double[] yArr = yList.ToArray();

                    if (xArr.Length <= 1)
                    {
                        writer.WriteLine("Недостаточно данных для анализа!");
                        writer.WriteLine();
                        continue;
                    }

                        RegressionPipelineOrchestrator.ProcessOneVarRegression(Path.GetFileName(excelPath), xArr, yArr, writer, runtimeConfig,
                            benchmarkResults);

                        writer.WriteLine();
                }
                catch (Exception ex)
                {
                    writer.WriteLine($"Ошибка при обработке файла: {ex.Message}");
                }
                writer.WriteLine();
            }

            BenchmarkReporting.WriteBenchmarkSummary(writer, benchmarkResults, runtimeConfig);
        }

        Console.WriteLine("Готово! Результаты записаны в " + outputPath);
        }


        internal static double CalculateRmseForLinearModel(List<double[]> data, double[] coefficients)
        {
            int n = data.Count;
            int p = data[0].Length - 1;
            double sse = 0;
            for (int i = 0; i < n; i++)
            {
                double predicted = coefficients[0];
                for (int j = 0; j < p; j++)
                    predicted += coefficients[j + 1] * data[i][j];

                double diff = data[i][p] - predicted;
                sse += diff * diff;
            }
            return Math.Sqrt(sse / n);
        }

        
        // Вычисление коэффициентов множественной регрессии
        internal static double[] CalculateMultipleRegressionCoefficients(List<double[]> data)
        {
            int n = data.Count;
            int p = data[0].Length - 1; // количество предикторов

            var x = new double[n][];
            var y = new double[n];
            for (int i = 0; i < n; i++)
            {
                var row = new double[p];
                for (int j = 0; j < p; j++)
                {
                    row[j] = data[i][j];
                }
                x[i] = row;
                y[i] = data[i][p];
            }

            return MultipleRegression.QR(x, y, intercept: true);
        }
        
        // Вычисление R²
        internal static double CalculateRSquared(List<double[]> data, double[] coefficients)
        {
            int n = data.Count;
            int p = data[0].Length - 1;
            
            double ssRes = 0; // Сумма квадратов остатков
            double ssTot = 0; // Общая сумма квадратов
            
            double meanY = data.Average(row => row[p]);
            
            for (int i = 0; i < n; i++)
            {
                double predicted = coefficients[0]; // Константа
                for (int j = 0; j < p; j++)
                {
                    predicted += coefficients[j + 1] * data[i][j];
                }
                
                double actual = data[i][p];
                double res = actual - predicted;
                double tot = actual - meanY;
                ssRes += res * res;
                ssTot += tot * tot;
            }
            
            return ssTot > 0 ? 1 - (ssRes / ssTot) : 1.0;
        }
        
        // Транспонирование матрицы
        private static double[,] TransposeMatrix(double[,] matrix, int rows, int cols)
        {
            var result = new double[cols, rows];
            for (int i = 0; i < rows; i++)
            {
                for (int j = 0; j < cols; j++)
                {
                    result[j, i] = matrix[i, j];
                }
            }
            return result;
        }
        
        // Умножение матриц
        private static double[,] MultiplyMatrices(double[,] A, double[,] B, int rowsA, int colsA, int colsB)
        {
            var result = new double[rowsA, colsB];
            for (int i = 0; i < rowsA; i++)
            {
                for (int j = 0; j < colsB; j++)
                {
                    for (int k = 0; k < colsA; k++)
                    {
                        result[i, j] += A[i, k] * B[k, j];
                    }
                }
            }
            return result;
        }
        
        // Умножение матрицы на вектор
        private static double[] MultiplyMatrixVector(double[,] matrix, double[] vector, int rows, int cols)
        {
            var result = new double[rows];
            for (int i = 0; i < rows; i++)
            {
                for (int j = 0; j < cols; j++)
                {
                    result[i] += matrix[i, j] * vector[j];
                }
            }
            return result;
        }
        
        // Решение системы линейных уравнений методом Гаусса
        private static double[] SolveLinearSystem(double[,] A, double[] b, int n)
        {
            var x = new double[n];
            var augmented = new double[n, n + 1];
            
            // Создаем расширенную матрицу
                        for (int i = 0; i < n; i++)
                        {
                for (int j = 0; j < n; j++)
                {
                    augmented[i, j] = A[i, j];
                }
                augmented[i, n] = b[i];
            }
            
            // Прямой ход метода Гаусса
            for (int i = 0; i < n; i++)
            {
                // Поиск максимального элемента в столбце
                int maxRow = i;
                for (int k = i + 1; k < n; k++)
                {
                    if (Math.Abs(augmented[k, i]) > Math.Abs(augmented[maxRow, i]))
                    {
                        maxRow = k;
                    }
                }
                
                // Обмен строк
                for (int k = i; k <= n; k++)
                {
                    double temp = augmented[i, k];
                    augmented[i, k] = augmented[maxRow, k];
                    augmented[maxRow, k] = temp;
                }
                
                // Приведение к треугольному виду
                for (int k = i + 1; k < n; k++)
                {
                    double factor = augmented[k, i] / augmented[i, i];
                    for (int j = i; j <= n; j++)
                    {
                        augmented[k, j] -= factor * augmented[i, j];
                    }
                }
            }
            
            // Обратный ход
            for (int i = n - 1; i >= 0; i--)
            {
                x[i] = augmented[i, n];
                for (int j = i + 1; j < n; j++)
                {
                    x[i] -= augmented[i, j] * x[j];
                }
                x[i] /= augmented[i, i];
            }
            
            return x;
        }
        
        // Функция для вычисления простых формул
        private static double EvaluateSimpleFormula(string formula, double x1, double x2)
        {
            try
            {
                switch (formula)
                {
                    case "x2 * x1":
                        return x2 * x1;
                    case "x2 * x1²":
                        return x2 * x1 * x1;
                    case "x2 * (x1 + x1²)":
                        return x2 * (x1 + x1 * x1);
                    case "x2 * exp(x1)":
                        return x2 * Math.Exp(x1);
                    case "x2 * sin(x1)":
                        return x2 * Math.Sin(x1);
                    case "x2 * (1 + x1)":
                        return x2 * (1 + x1);
                    case "x2 * (0.1 + x1 + x1²)":
                        return x2 * (0.1 + x1 + x1 * x1);
                    case "sin(x2)² * exp(x1)":
                        return Math.Pow(Math.Sin(x2), 2) * Math.Exp(x1);
                    case "sin(x2)² * exp(0.5*x1)":
                        return Math.Pow(Math.Sin(x2), 2) * Math.Exp(0.5 * x1);
                    case "sin(x2)² * exp(0.1*x1)":
                        return Math.Pow(Math.Sin(x2), 2) * Math.Exp(0.1 * x1);
                    case "sin(x2)² * exp(2*x1)":
                        return Math.Pow(Math.Sin(x2), 2) * Math.Exp(2 * x1);
                    case "sin(x2) * exp(x1)":
                        return Math.Sin(x2) * Math.Exp(x1);
                    case "sin(x2) * exp(0.5*x1)":
                        return Math.Sin(x2) * Math.Exp(0.5 * x1);
                    case "sin(x2)² * exp(0.2*x1)":
                        return Math.Pow(Math.Sin(x2), 2) * Math.Exp(0.2 * x1);
                    case "sin(x2)² * exp(0.3*x1)":
                        return Math.Pow(Math.Sin(x2), 2) * Math.Exp(0.3 * x1);
                    case "sin(x2)² * exp(0.4*x1)":
                        return Math.Pow(Math.Sin(x2), 2) * Math.Exp(0.4 * x1);
                    case "sin(x2)² * exp(0.6*x1)":
                        return Math.Pow(Math.Sin(x2), 2) * Math.Exp(0.6 * x1);
                    case "sin(x2)² * exp(0.7*x1)":
                        return Math.Pow(Math.Sin(x2), 2) * Math.Exp(0.7 * x1);
                    case "sin(x2)² * exp(0.8*x1)":
                        return Math.Pow(Math.Sin(x2), 2) * Math.Exp(0.8 * x1);
                    case "sin(x2)² * exp(0.9*x1)":
                        return Math.Pow(Math.Sin(x2), 2) * Math.Exp(0.9 * x1);
                    case "sin(x2)² * exp(1.5*x1)":
                        return Math.Pow(Math.Sin(x2), 2) * Math.Exp(1.5 * x1);
                    case "sin(x2)² * exp(3*x1)":
                        return Math.Pow(Math.Sin(x2), 2) * Math.Exp(3 * x1);
                    default:
                        return double.NaN;
                }
            }
            catch
            {
                return double.NaN;
            }
        }
        
        // Вспомогательная функция для вычисления значения дерева
        static double EvaluateTree(ExpressionNode node, double x)
        {
            switch (node.Operation)
            {
                case "const":
                    return node.Constant ?? 0;
                case "x":
                    return x;
                case "x2":
                    return x * x;
                case "x3":
                    return x * x * x;
                case "+":
                    return EvaluateTree(node.Children[0], x) + EvaluateTree(node.Children[1], x);
                case "-":
                    return EvaluateTree(node.Children[0], x) - EvaluateTree(node.Children[1], x);
                case "*":
                    return EvaluateTree(node.Children[0], x) * EvaluateTree(node.Children[1], x);
                case "/":
                    var denom = EvaluateTree(node.Children[1], x);
                    return denom == 0 ? 0 : EvaluateTree(node.Children[0], x) / denom;
                case "sin":
                    return Math.Sin(EvaluateTree(node.Children[0], x));
                case "cos":
                    return Math.Cos(EvaluateTree(node.Children[0], x));
                case "exp":
                    var expArg = EvaluateTree(node.Children[0], x);
                    return expArg > 100 ? 1e100 : Math.Exp(expArg);
                case "log":
                    var logArg = EvaluateTree(node.Children[0], x);
                    return logArg <= 0 ? -1000 : Math.Log(logArg);
                case "pow":
                    var baseVal = EvaluateTree(node.Children[0], x);
                    var power = EvaluateTree(node.Children[1], x);
                    return Math.Pow(baseVal, power);
                default:
                    return 0;
            }
        }
        
        // Вспомогательная функция для вычисления значения дерева с множественными переменными
        static double EvaluateTree(ExpressionNode node, double x1, double x2)
        {
            switch (node.Operation)
            {
                case "const":
                    return node.Constant ?? 0;
                case "x":
                    return x1;
                case "x_2":
                    return x2;
                case "x1²":
                    return x1 * x1;
                case "x_2²":
                    return x2 * x2;
                case "x1³":
                    return x1 * x1 * x1;
                case "x_2³":
                    return x2 * x2 * x2;
                case "+":
                    return EvaluateTree(node.Children[0], x1, x2) + EvaluateTree(node.Children[1], x1, x2);
                case "-":
                    return EvaluateTree(node.Children[0], x1, x2) - EvaluateTree(node.Children[1], x1, x2);
                case "*":
                    return EvaluateTree(node.Children[0], x1, x2) * EvaluateTree(node.Children[1], x1, x2);
                case "/":
                    var denom = EvaluateTree(node.Children[1], x1, x2);
                    return denom == 0 ? 0 : EvaluateTree(node.Children[0], x1, x2) / denom;
                case "sin":
                    return Math.Sin(EvaluateTree(node.Children[0], x1, x2));
                case "cos":
                    return Math.Cos(EvaluateTree(node.Children[0], x1, x2));
                case "exp":
                    var expArg = EvaluateTree(node.Children[0], x1, x2);
                    return expArg > 100 ? 1e100 : Math.Exp(expArg);
                case "log":
                    var logArg = EvaluateTree(node.Children[0], x1, x2);
                    return logArg <= 0 ? -1000 : Math.Log(logArg);
                case "pow":
                    var baseVal = EvaluateTree(node.Children[0], x1, x2);
                    var power = EvaluateTree(node.Children[1], x1, x2);
                    return Math.Pow(baseVal, power);
                default:
                    return 0;
            }
        }

        // Универсальная оценка дерева для n переменных, поддерживает
        internal static ExpressionNode ExpandAndSimplifyBaseFormula(ExpressionNode root, List<ExpressionNode>? baseFunctions)
        {
            var expanded = ExpandBaseReferences(root, baseFunctions, new HashSet<int>());
            var simplified = SimplifyExpression(expanded);
            return RewriteSinCosLinearCombination(simplified);
        }

        /// true, если в дереве есть узлы base (ссылки f1..fk в режиме суперпозиции)
        internal static bool ExpressionTreeUsesBaseCalls(ExpressionNode? node)
        {
            if (node == null) return false;
            if (string.Equals(node.Operation, "base", StringComparison.Ordinal))
                return true;
            foreach (var ch in node.Children)
            {
                if (ExpressionTreeUsesBaseCalls(ch))
                    return true;
            }
            return false;
        }

        // RMSE одного-переменного дерева (с защитой от NaN/Inf) невалидные предсказания пропускаются, что позволяет корректно сравнивать
        internal static double ComputeRmseSingleVar(ExpressionNode tree, double[] xArr, double[] yArr)
        {
            double sse = 0;
            int valid = 0;
            for (int i = 0; i < xArr.Length; i++)
            {
                double predicted = EvaluateTreeN(tree, new[] { xArr[i] }, null);
                if (double.IsNaN(predicted) || double.IsInfinity(predicted)) continue;
                double diff = yArr[i] - predicted;
                sse += diff * diff;
                valid++;
            }
            if (valid == 0) return double.MaxValue;
            return Math.Sqrt(sse / valid);
        }

        // RMSE дерева на (X, y) для произвольного количества переменных,с поддержкой базовых функций (могут быть null для чистых baseline-деревьев).
        internal static double ComputeRmseN(ExpressionNode tree, double[][] X, double[] y, List<ExpressionNode>? baseFunctions)
        {
            double sse = 0;
            int valid = 0;
            for (int i = 0; i < X.Length; i++)
            {
                double predicted = EvaluateTreeN(tree, X[i], baseFunctions);
                if (double.IsNaN(predicted) || double.IsInfinity(predicted)) continue;
                double diff = y[i] - predicted;
                sse += diff * diff;
                valid++;
            }
            return valid > 0 ? Math.Sqrt(sse / valid) : double.MaxValue;
        }

        internal static bool NearlyCollinearPredictors(double[][] X, int p, out string? note)
        {
            note = null;
            if (p != 2 || X.Length < 5) return false;
            int n = X.Length;
            var xs = new double[n][];
            for (int i = 0; i < n; i++)
                xs[i] = new[] { X[i][0] };
            var x2col = new double[n];
            for (int i = 0; i < n; i++)
                x2col[i] = X[i][1];
            try
            {
                var cf = MultipleRegression.QR(xs, x2col, intercept: true);
                double a = cf[0], b = cf[1];
                double span = x2col.Max() - x2col.Min();
                double maxR = 0;
                for (int i = 0; i < n; i++)
                    maxR = Math.Max(maxR, Math.Abs(x2col[i] - (a + b * X[i][0])));
                if (maxR > Math.Max(1e-9, 1e-9 * Math.Max(span, 1e-12)))
                    return false;
                note =
                    $"X2 почти лежит на прямой от X1 (x2 ≈ {a:F4} + {b:F4}·x1). " +
                    "Линейные коэффициенты при x1, x2 для одной и той же зависимости в плоскости (x1,x2) не единственны; " +
                    "ниже — одно из целочисленных эквивалентных представлений с минимальной «сложностью» (если найдено).";
                return true;
            }
            catch
            {
                return false;
            }
        }

        internal static ExpressionNode? BuildLinearBaseline(double[][] X, double[] y, int p)
        {
            if (X.Length < p + 2) return null;
            try
            {
                var coef = LinearRegressionMnls(X, y, p);
                if (p == 2 && TryBestIntegerLinearExact(X, y, p, out var intTree))
                    return intTree;
                return BuildLinearTree(coef, p);
            }
            catch { return null; }
        }

        static double[] LinearRegressionMnls(double[][] X, double[] y, int p)
        {
            int n = X.Length;
            var A = Matrix<double>.Build.Dense(n, p + 1, (i, j) => j == 0 ? 1.0 : X[i][j - 1]);
            var yv = Vector<double>.Build.Dense(y);
            var pinv = A.PseudoInverse();
            return pinv.Multiply(yv).ToArray();
        }

        static bool TryBestIntegerLinearExact(double[][] X, double[] y, int p, out ExpressionNode tree)
        {
            tree = null!;
            int n = X.Length;
            if (p < 1 || p > 3) return false;
            double yRange = y.Max() - y.Min();
            double tol = Math.Max(1e-10 * Math.Max(yRange, 1e-12), 1e-12);
            const int R = 42;
            int bestSumSq = int.MaxValue;
            int bestAbsW0 = int.MaxValue;
            var bestW = new int[p + 1];
            bool found = false;

            if (p == 1)
            {
                for (int w0 = -R; w0 <= R; w0++)
                    for (int w1 = -R; w1 <= R; w1++)
                    {
                        double maxE = 0;
                        for (int i = 0; i < n; i++)
                            maxE = Math.Max(maxE, Math.Abs(y[i] - (w0 + w1 * X[i][0])));
                        if (maxE > tol) continue;
                        found = true;
                        int s = w1 * w1;
                        int aw0 = Math.Abs(w0);
                        if (s < bestSumSq || (s == bestSumSq && aw0 < bestAbsW0))
                        {
                            bestSumSq = s;
                            bestAbsW0 = aw0;
                            bestW[0] = w0;
                            bestW[1] = w1;
                        }
                    }
            }
            else if (p == 2)
            {
                for (int w0 = -R; w0 <= R; w0++)
                    for (int w1 = -R; w1 <= R; w1++)
                        for (int w2 = -R; w2 <= R; w2++)
                        {
                            double maxE = 0;
                            for (int i = 0; i < n; i++)
                                maxE = Math.Max(maxE,
                                    Math.Abs(y[i] - (w0 + w1 * X[i][0] + w2 * X[i][1])));
                            if (maxE > tol) continue;
                            found = true;
                            int s = w1 * w1 + w2 * w2;
                            int aw0 = Math.Abs(w0);
                            if (s < bestSumSq || (s == bestSumSq && aw0 < bestAbsW0))
                            {
                                bestSumSq = s;
                                bestAbsW0 = aw0;
                                bestW[0] = w0;
                                bestW[1] = w1;
                                bestW[2] = w2;
                            }
                        }
            }
            else
            {
                // p==3 полный перебор был бы слишком большим — пропускаем
                return false;
            }

            if (!found) return false;
            tree = BuildLinearTreeFromInts(bestW, p);
            return true;
        }

        static ExpressionNode BuildLinearTreeFromInts(int[] w, int p)
        {
            var coef = new double[p + 1];
            for (int i = 0; i <= p; i++) coef[i] = w[i];
            return BuildLinearTree(coef, p);
        }

        // Квадратичный baseline (включая чистые квадраты xi²). Покрывает поверхности 2-го порядка.
        internal static ExpressionNode? BuildQuadraticBaseline(double[][] X, double[] y, int p)
        {
            int n = X.Length;
            int qFeats = p + p * (p + 1) / 2;
            if (n < qFeats + 2) return null;

            try
            {
                var XAug = new double[n][];
                for (int i = 0; i < n; i++)
                {
                    var row = new double[qFeats];
                    int k = 0;
                    for (int j = 0; j < p; j++) row[k++] = X[i][j];
                    for (int j1 = 0; j1 < p; j1++)
                        for (int j2 = j1; j2 < p; j2++)
                            row[k++] = X[i][j1] * X[i][j2];
                    XAug[i] = row;
                }
                var coef = MultipleRegression.QR(XAug, y, intercept: true);
                return BuildQuadraticTree(coef, p);
            }
            catch { return null; }
        }

        internal static ExpressionNode? BuildLogLinearBaseline(double[][] X, double[] y, int p)
        {
            int n = X.Length;
            if (n < p + 2) return null;
            for (int i = 0; i < n; i++) if (y[i] <= 0) return null;

            try
            {
                var logY = new double[n];
                for (int i = 0; i < n; i++) logY[i] = Math.Log(y[i]);
                var coef = MultipleRegression.QR(X, logY, intercept: true);
                var inner = BuildLinearTree(coef, p);
                return new ExpressionNode { Operation = "exp", Children = { inner } };
            }
            catch { return null; }
        }

        static ExpressionNode BuildLinearTree(double[] coef, int p)
        {
            ExpressionNode? result = null;
            if (Math.Abs(coef[0]) > 1e-12)
                result = new ExpressionNode { Operation = "const", Constant = coef[0] };

            for (int i = 0; i < p; i++)
            {
                if (Math.Abs(coef[i + 1]) < 1e-12) continue;
                var term = new ExpressionNode
                {
                    Operation = "*",
                    Children =
                    {
                        new ExpressionNode { Operation = "const", Constant = coef[i + 1] },
                        new ExpressionNode { Operation = "var", VariableIndex = i }
                    }
                };
                result = result == null ? term : new ExpressionNode { Operation = "+", Children = { result, term } };
            }
            return result ?? new ExpressionNode { Operation = "const", Constant = 0 };
        }

        static ExpressionNode BuildQuadraticTree(double[] coef, int p)
        {
            var terms = new List<ExpressionNode>();

            if (Math.Abs(coef[0]) > 1e-12)
                terms.Add(new ExpressionNode { Operation = "const", Constant = coef[0] });

            int idx = 1;
            for (int j = 0; j < p; j++, idx++)
            {
                if (Math.Abs(coef[idx]) < 1e-12) continue;
                terms.Add(new ExpressionNode
                {
                    Operation = "*",
                    Children =
                    {
                        new ExpressionNode { Operation = "const", Constant = coef[idx] },
                        new ExpressionNode { Operation = "var", VariableIndex = j }
                    }
                });
            }

            for (int j1 = 0; j1 < p; j1++)
            {
                for (int j2 = j1; j2 < p; j2++, idx++)
                {
                    if (Math.Abs(coef[idx]) < 1e-12) continue;
                    var prod = new ExpressionNode
                    {
                        Operation = "*",
                        Children =
                        {
                            new ExpressionNode { Operation = "var", VariableIndex = j1 },
                            new ExpressionNode { Operation = "var", VariableIndex = j2 }
                        }
                    };
                    terms.Add(new ExpressionNode
                    {
                        Operation = "*",
                        Children =
                        {
                            new ExpressionNode { Operation = "const", Constant = coef[idx] },
                            prod
                        }
                    });
                }
            }

            if (terms.Count == 0) return new ExpressionNode { Operation = "const", Constant = 0 };

            var result = terms[0];
            for (int i = 1; i < terms.Count; i++)
                result = new ExpressionNode { Operation = "+", Children = { result, terms[i] } };
            return result;
        }

        static ExpressionNode RewriteSinCosLinearCombination(ExpressionNode node)
        {
            for (int i = 0; i < node.Children.Count; i++)
                node.Children[i] = RewriteSinCosLinearCombination(node.Children[i]);

            if (node.Operation != "+" || node.Children.Count != 2) return node;

            var leftMatch = TryMatchAmplitudeFunc(node.Children[0]);
            var rightMatch = TryMatchAmplitudeFunc(node.Children[1]);
            if (leftMatch == null || rightMatch == null) return node;
            if (leftMatch.Value.Func == rightMatch.Value.Func) return node;

            bool leftIsSin = leftMatch.Value.Func == "sin";
            var sinPart = leftIsSin ? leftMatch.Value : rightMatch.Value;
            var cosPart = leftIsSin ? rightMatch.Value : leftMatch.Value;

            if (!StructuralEqualsApprox(sinPart.Arg, cosPart.Arg, 0.01)) return node;

            double a = sinPart.Amp;
            double b = cosPart.Amp;
            double A = Math.Sqrt(a * a + b * b);
            if (A < 1e-12) return node;
            double phi = Math.Atan2(b, a);

            var arg = sinPart.Arg.Clone();
            ExpressionNode argPlusPhi = Math.Abs(phi) < 1e-9
                ? arg
                : new ExpressionNode
                {
                    Operation = "+",
                    Children = { arg, new ExpressionNode { Operation = "const", Constant = phi } }
                };

            var sinNode = new ExpressionNode
            {
                Operation = "sin",
                Children = { argPlusPhi }
            };

            if (Math.Abs(A - 1.0) < 1e-9) return sinNode;
            return new ExpressionNode
            {
                Operation = "*",
                Children = { new ExpressionNode { Operation = "const", Constant = A }, sinNode }
            };
        }

        private readonly struct AmpFuncMatch
        {
            public AmpFuncMatch(string func, double amp, ExpressionNode arg)
            {
                Func = func;
                Amp = amp;
                Arg = arg;
            }
            public string Func { get; }
            public double Amp { get; }
            public ExpressionNode Arg { get; }
        }


        static AmpFuncMatch? TryMatchAmplitudeFunc(ExpressionNode node)
        {
            if ((node.Operation == "sin" || node.Operation == "cos") && node.Children.Count == 1)
                return new AmpFuncMatch(node.Operation, 1.0, node.Children[0]);

            if (node.Operation == "*" && node.Children.Count == 2)
            {
                var c0 = node.Children[0];
                var c1 = node.Children[1];

                if (c0.Operation == "const" && (c1.Operation == "sin" || c1.Operation == "cos") && c1.Children.Count == 1)
                    return new AmpFuncMatch(c1.Operation, c0.Constant ?? 0, c1.Children[0]);

                if (c1.Operation == "const" && (c0.Operation == "sin" || c0.Operation == "cos") && c0.Children.Count == 1)
                    return new AmpFuncMatch(c0.Operation, c1.Constant ?? 0, c0.Children[0]);
            }

            return null;
        }

        // Структурное равенство с допуском по числовым константам.
        static bool StructuralEqualsApprox(ExpressionNode a, ExpressionNode b, double relTol)
        {
            if (a.Operation != b.Operation) return false;
            if (a.VariableIndex != b.VariableIndex) return false;
            if (a.BaseFunctionIndex != b.BaseFunctionIndex) return false;

            if (a.Operation == "const")
            {
                double av = a.Constant ?? 0, bv = b.Constant ?? 0;
                double denom = Math.Max(Math.Abs(av), Math.Max(Math.Abs(bv), 1e-9));
                return Math.Abs(av - bv) / denom <= relTol;
            }

            if (a.Children.Count != b.Children.Count) return false;
            for (int i = 0; i < a.Children.Count; i++)
                if (!StructuralEqualsApprox(a.Children[i], b.Children[i], relTol)) return false;
            return true;
        }

        static ExpressionNode ExpandBaseReferences(ExpressionNode node, List<ExpressionNode>? baseFunctions, HashSet<int> stack)
        {
            if (node.Operation == "base" && baseFunctions != null)
            {
                int idx = node.BaseFunctionIndex ?? -1;
                if (idx >= 0 && idx < baseFunctions.Count)
                {
                    if (stack.Contains(idx))
                    {
                        return node.Clone();
                    }

                    stack.Add(idx);
                    var expandedBase = ExpandBaseReferences(baseFunctions[idx], baseFunctions, stack);
                    stack.Remove(idx);
                    return expandedBase;
                }
            }

            var clone = node.Clone();
            for (int i = 0; i < clone.Children.Count; i++)
            {
                clone.Children[i] = ExpandBaseReferences(clone.Children[i], baseFunctions, stack);
            }
            return clone;
        }

        static ExpressionNode SimplifyExpression(ExpressionNode node)
        {
            if (node.Children.Count > 0)
            {
                for (int i = 0; i < node.Children.Count; i++)
                    node.Children[i] = SimplifyExpression(node.Children[i]);
            }

            if (node.Operation is "+" or "-" or "*" or "/" or "pow")
            {
                if (node.Children.Count >= 2 &&
                    node.Children[0].Operation == "const" &&
                    node.Children[1].Operation == "const")
                {
                    double a = node.Children[0].Constant ?? 0;
                    double b = node.Children[1].Constant ?? 0;
                    return node.Operation switch
                    {
                        "+" => new ExpressionNode { Operation = "const", Constant = a + b },
                        "-" => new ExpressionNode { Operation = "const", Constant = a - b },
                        "*" => new ExpressionNode { Operation = "const", Constant = a * b },
                        "/" => Math.Abs(b) < 1e-12 ? node : new ExpressionNode { Operation = "const", Constant = a / b },
                        "pow" => new ExpressionNode { Operation = "const", Constant = Math.Pow(a, b) },
                        _ => node
                    };
                }
            }

            if (node.Operation == "+" && node.Children.Count == 2)
            {
                if (IsConst(node.Children[0], 0)) return node.Children[1];
                if (IsConst(node.Children[1], 0)) return node.Children[0];
            }
            if (node.Operation == "-" && node.Children.Count == 2)
            {
                if (IsConst(node.Children[1], 0)) return node.Children[0];
            }
            if (node.Operation == "*" && node.Children.Count == 2)
            {
                if (IsConst(node.Children[0], 0) || IsConst(node.Children[1], 0))
                    return new ExpressionNode { Operation = "const", Constant = 0 };
                if (IsConst(node.Children[0], 1)) return node.Children[1];
                if (IsConst(node.Children[1], 1)) return node.Children[0];
            }
            if (node.Operation == "/" && node.Children.Count == 2)
            {
                if (IsConst(node.Children[0], 0)) return new ExpressionNode { Operation = "const", Constant = 0 };
                if (IsConst(node.Children[1], 1)) return node.Children[0];
            }

            if (node.Operation == "+" && node.Children.Count == 2)
            {
                var left = node.Children[0];
                var right = node.Children[1];
                if (left.Operation == "-" && left.Children.Count == 2 && right.Operation == "+" && right.Children.Count == 2)
                {
                    if (StructuralEquals(left.Children[1], right.Children[0]))
                    {
                        return SimplifyExpression(new ExpressionNode
                        {
                            Operation = "+",
                            Children = { left.Children[0], right.Children[1] }
                        });
                    }
                    if (StructuralEquals(left.Children[1], right.Children[1]))
                    {
                        return SimplifyExpression(new ExpressionNode
                        {
                            Operation = "+",
                            Children = { left.Children[0], right.Children[0] }
                        });
                    }
                }
            }

            return node;
        }

        static bool IsConst(ExpressionNode node, double value)
        {
            return node.Operation == "const" && Math.Abs((node.Constant ?? 0) - value) < 1e-12;
        }

        static bool StructuralEquals(ExpressionNode a, ExpressionNode b)
        {
            if (a.Operation != b.Operation) return false;
            if (a.VariableIndex != b.VariableIndex) return false;
            if (a.BaseFunctionIndex != b.BaseFunctionIndex) return false;
            if ((a.Constant.HasValue || b.Constant.HasValue) &&
                Math.Abs((a.Constant ?? 0) - (b.Constant ?? 0)) > 1e-12)
                return false;
            if (a.Children.Count != b.Children.Count) return false;
            for (int i = 0; i < a.Children.Count; i++)
            {
                if (!StructuralEquals(a.Children[i], b.Children[i])) return false;
            }
            return true;
        }

        internal static double EvaluateTreeN(ExpressionNode node, double[] vars, List<ExpressionNode>? baseFunctions = null)
        {
            switch (node.Operation)
            {
                case "const":
                    return node.Constant ?? 0;
                case "var":
                {
                    int idx = node.VariableIndex ?? 0;
                    return (idx >= 0 && idx < vars.Length) ? vars[idx] : double.NaN;
                }
                case "base":
                {
                    if (baseFunctions == null) return double.NaN;
                    int idx = node.BaseFunctionIndex ?? -1;
                    if (idx < 0 || idx >= baseFunctions.Count) return double.NaN;
                    return EvaluateTreeN(baseFunctions[idx], vars, baseFunctions);
                }
                case "neg":
                {
                    if (node.Children.Count < 1) return double.NaN;
                    double v = EvaluateTreeN(node.Children[0], vars, baseFunctions);
                    return (double.IsNaN(v) || double.IsInfinity(v)) ? double.NaN : -v;
                }
                case "+":
                case "-":
                case "*":
                case "/":
                {
                    if (node.Children.Count < 2) return double.NaN;
                    double a = EvaluateTreeN(node.Children[0], vars, baseFunctions);
                    double b = EvaluateTreeN(node.Children[1], vars, baseFunctions);
                    if (double.IsNaN(a) || double.IsNaN(b) || double.IsInfinity(a) || double.IsInfinity(b))
                        return double.NaN;
                    return node.Operation switch
                    {
                        "+" => a + b,
                        "-" => a - b,
                        "*" => a * b,
                        "/" => Math.Abs(b) < 1e-10 ? double.NaN : a / b,
                        _ => double.NaN
                    };
                }
                case "sin":
                case "cos":
                case "exp":
                case "log":
                {
                    if (node.Children.Count < 1) return double.NaN;
                    double a = EvaluateTreeN(node.Children[0], vars, baseFunctions);
                    if (double.IsNaN(a) || double.IsInfinity(a)) return double.NaN;
                    return node.Operation switch
                    {
                        "sin" => Math.Sin(a),
                        "cos" => Math.Cos(a),
                        "exp" => a > 709 ? double.PositiveInfinity : Math.Exp(a),
                        "log" => a <= 0 ? double.NaN : Math.Log(a),
                        _ => double.NaN
                    };
                }
                case "pow":
                {
                    if (node.Children.Count < 2) return double.NaN;
                    double a = EvaluateTreeN(node.Children[0], vars, baseFunctions);
                    double p = EvaluateTreeN(node.Children[1], vars, baseFunctions);
                    if (double.IsNaN(a) || double.IsNaN(p) || double.IsInfinity(a) || double.IsInfinity(p))
                        return double.NaN;
                    if (Math.Abs(p) > 10) return double.NaN;
                    if (a < 0 && p != Math.Floor(p)) return double.NaN;
                    return Math.Pow(a, p);
                }
                default:
                    return double.NaN;
            }
        }
    }
}
