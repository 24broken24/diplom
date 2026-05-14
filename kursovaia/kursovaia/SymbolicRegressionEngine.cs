using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Text;
using System.Text.RegularExpressions;
using MathNet.Numerics;
using MathNet.Numerics.LinearAlgebra;
using MathNet.Numerics.Optimization;

namespace SymbolicRegression
{
    public class SymbolicRegressionEngine
    {
        /// <summary>Параметры журнала эволюции; задаётся из gp_settings.json перед запуском.</summary>
        public static GpEvolutionTraceConfig? GlobalEvolutionTrace { get; set; }

        /// <summary>Метка запуска для строк лога (например, имя файла и режим direct/log-fit/multi-var).</summary>
        public static string EvolutionTraceRunLabel { get; set; } = "";

        /// <summary>Базовый seed рестартов (см. gp_settings.json → gpRandomBaseSeed). Игнорируется при GpRandomizeBaseEachRun.</summary>
        public static int GpRandomBaseSeed { get; set; } = 12345;

        /// <summary>Случайная база перед каждым Evolve() (gp_settings.json → gpRandomizeBaseEachRun).</summary>
        public static bool GpRandomizeBaseEachRun { get; set; }

        private static readonly object EvolutionTraceLock = new();
        public enum GrammarProfile
        {
            Linear,
            Polynomial,
            Trigonometric,
            General,
            Superposition
        }

        private sealed class GpConfig
        {
            public int MaxDepth { get; init; } = 5;
            public int PopulationSize { get; init; } = 220;
            public int Generations { get; init; } = 120;
            public int Restarts { get; init; } = 8;
            public double MutationRate { get; init; } = 0.15;
            public double CrossoverRate { get; init; } = 0.8;
            public double TerminalConstProbability { get; init; } = 0.35;
            public double ComplexityPenalty { get; init; } = 0.01;
            public double InvalidPredictionPenalty { get; init; } = 10_000.0;
            public double MaxExpArg { get; init; } = 20.0;
            public double MaxPowAbs { get; init; } = 8.0;
            public string[] AllowedOps { get; init; } = new[] { "+", "-", "*", "/", "sin", "cos", "exp", "log", "pow", "neg" };
        }

        private Random random = new Random();
        private double[] xData;
        private double[] yData;
        private double[][]? XData; // [row][var] for n-variables
        private int variableCount = 1;
        private int maxDepth = 6;
        private int populationSize = 300;
        private int generations = 150;
        private double mutationRate = 0.15;
        private double crossoverRate = 0.8;
        private readonly List<ExpressionNode> baseFunctions = new List<ExpressionNode>();
        private readonly bool superpositionMode;
        private readonly GpConfig config;
        private static readonly Dictionary<string, GpProfileOverride> ExternalProfileOverrides = new(StringComparer.OrdinalIgnoreCase);

        public static void SetExternalProfileOverrides(Dictionary<string, GpProfileOverride>? profiles)
        {
            ExternalProfileOverrides.Clear();
            if (profiles == null) return;
            foreach (var kv in profiles)
                ExternalProfileOverrides[kv.Key] = kv.Value;
        }
        
        public SymbolicRegressionEngine(double[] x, double[] y)
        {
            xData = x;
            yData = y;
            AnalyzeDataPattern();
            variableCount = 1;
            superpositionMode = false;
            config = BuildConfig(MapPatternToProfile(dataPattern));
            ApplyConfig();
        }

        // Перегрузка для случая, когда внешний код хочет принудительно выбрать
        // профиль грамматики (например, для log-fit мы знаем, что в log-пространстве
        // нужны все операции — берём General независимо от паттерна).
        public SymbolicRegressionEngine(double[] x, double[] y, GrammarProfile forcedProfile)
        {
            xData = x;
            yData = y;
            AnalyzeDataPattern();
            variableCount = 1;
            superpositionMode = false;
            config = BuildConfig(forcedProfile);
            // Помечаем dataPattern из принудительного профиля для информативного вывода
            dataPattern = forcedProfile switch
            {
                GrammarProfile.Linear => "linear",
                GrammarProfile.Polynomial => "polynomial",
                GrammarProfile.Trigonometric => "trigonometric",
                GrammarProfile.Superposition => "superposition",
                _ => "general"
            };
            ApplyConfig();
        }

        /// <summary>
        /// Multi-var: матрица наблюдений <paramref name="X"/> (строка — объект, столбцы — x₁…xₚ). Единственный путь n переменных в движке.
        /// </summary>
        public SymbolicRegressionEngine(double[][] X, double[] y, List<ExpressionNode>? userBaseFunctions = null, bool superposition = true)
        {
            XData = X;
            yData = y;
            // Первая колонка X — для одномерных эвристик (если когда-нибудь понадобятся поверх n-var)
            xData = X.Length > 0 ? X.Select(r => r.Length > 0 ? r[0] : 0.0).ToArray() : Array.Empty<double>();
            variableCount = X.Length > 0 ? X[0].Length : 0;
            dataPattern = "multi-var";

            if (userBaseFunctions != null && userBaseFunctions.Count > 0)
                baseFunctions.AddRange(userBaseFunctions);

            superpositionMode = superposition && baseFunctions.Count > 0;

            maxDepth = superpositionMode ? 4 : 5;
            populationSize = 220;
            generations = 120;
            config = BuildConfig(superpositionMode ? GrammarProfile.Superposition : GrammarProfile.General);
            ApplyConfig();
        }

        private GrammarProfile MapPatternToProfile(string pattern) => pattern switch
        {
            "linear" => GrammarProfile.Linear,
            "polynomial" => GrammarProfile.Polynomial,
            "trigonometric" => GrammarProfile.Trigonometric,
            "superposition" => GrammarProfile.Superposition,
            _ => GrammarProfile.General
        };

        private GpConfig BuildConfig(GrammarProfile profile)
        {
            var baseConfig = profile switch
            {
                GrammarProfile.Linear => new GpConfig
                {
                    MaxDepth = 4,
                    PopulationSize = 180,
                    Generations = 90,
                    Restarts = 6,
                    AllowedOps = new[] { "+", "-", "*" },
                    ComplexityPenalty = 0.02,
                    MaxPowAbs = 2
                },
                GrammarProfile.Polynomial => new GpConfig
                {
                    MaxDepth = 5,
                    PopulationSize = 220,
                    Generations = 120,
                    Restarts = 8,
                    AllowedOps = new[] { "+", "-", "*", "pow" },
                    ComplexityPenalty = 0.015,
                    MaxPowAbs = 4
                },
                GrammarProfile.Trigonometric => new GpConfig
                {
                    MaxDepth = 5,
                    PopulationSize = 220,
                    Generations = 140,
                    Restarts = 10,
                    AllowedOps = new[] { "+", "-", "*", "sin", "cos", "neg" },
                    ComplexityPenalty = 0.015
                },
                GrammarProfile.Superposition => new GpConfig
                {
                    MaxDepth = 4,
                    PopulationSize = 240,
                    Generations = 140,
                    Restarts = 10,
                    AllowedOps = new[] { "+", "-", "*", "/", "sin", "cos", "exp", "log", "pow", "neg" },
                    ComplexityPenalty = 0.02,
                    MaxPowAbs = 4
                },
                _ => new GpConfig()
            };

            string key = profile switch
            {
                GrammarProfile.Linear => "linear",
                GrammarProfile.Polynomial => "polynomial",
                GrammarProfile.Trigonometric => "trigonometric",
                GrammarProfile.Superposition => "superposition",
                _ => "general"
            };

            if (ExternalProfileOverrides.TryGetValue(key, out var ov))
                return ApplyOverride(baseConfig, ov);

            return baseConfig;
        }

        private static GpConfig ApplyOverride(GpConfig baseCfg, GpProfileOverride ov)
        {
            return new GpConfig
            {
                MaxDepth = ov.MaxDepth ?? baseCfg.MaxDepth,
                PopulationSize = ov.PopulationSize ?? baseCfg.PopulationSize,
                Generations = ov.Generations ?? baseCfg.Generations,
                Restarts = ov.Restarts ?? baseCfg.Restarts,
                MutationRate = ov.MutationRate ?? baseCfg.MutationRate,
                CrossoverRate = ov.CrossoverRate ?? baseCfg.CrossoverRate,
                TerminalConstProbability = ov.TerminalConstProbability ?? baseCfg.TerminalConstProbability,
                ComplexityPenalty = ov.ComplexityPenalty ?? baseCfg.ComplexityPenalty,
                InvalidPredictionPenalty = ov.InvalidPredictionPenalty ?? baseCfg.InvalidPredictionPenalty,
                MaxExpArg = ov.MaxExpArg ?? baseCfg.MaxExpArg,
                MaxPowAbs = ov.MaxPowAbs ?? baseCfg.MaxPowAbs,
                AllowedOps = (ov.AllowedOps != null && ov.AllowedOps.Length > 0) ? ov.AllowedOps : baseCfg.AllowedOps
            };
        }

        private void ApplyConfig()
        {
            maxDepth = config.MaxDepth;
            populationSize = config.PopulationSize;
            generations = config.Generations;
            mutationRate = config.MutationRate;
            crossoverRate = config.CrossoverRate;
        }

        public static List<ExpressionNode> LoadBaseFunctions(string path, int variableCount, List<string>? warnings = null)
        {
            var list = new List<ExpressionNode>();
            if (!File.Exists(path)) return list;

            foreach (var raw in File.ReadAllLines(path))
            {
                var line = raw.Trim();
                if (line.Length == 0) continue;
                if (line.StartsWith("#")) continue;

                // формат: f1 = <expr> или просто <expr>
                int eq = line.IndexOf('=');
                string expr = eq >= 0 ? line[(eq + 1)..].Trim() : line;
                if (expr.Length == 0) continue;

                int maxVar = DetectMaxVariableIndex(expr);
                if (maxVar > variableCount)
                {
                    warnings?.Add($"Пропущена базовая функция '{line}': требуется x{maxVar}, а в данных только x1..x{variableCount}");
                    continue;
                }

                try
                {
                    var parser = new ExpressionParser(expr, variableCount);
                    list.Add(parser.Parse());
                }
                catch (Exception ex)
                {
                    // Не валим весь запуск из-за одной неподходящей базовой функции
                    warnings?.Add($"Пропущена базовая функция '{line}': {ex.Message}");
                }
            }
            return list;
        }

        private static int DetectMaxVariableIndex(string expr)
        {
            int maxIdx = 0;
            foreach (Match m in Regex.Matches(expr, @"\bx(\d+)\b", RegexOptions.IgnoreCase))
            {
                if (int.TryParse(m.Groups[1].Value, out int idx) && idx > maxIdx)
                    maxIdx = idx;
            }
            return maxIdx;
        }
        
        private string dataPattern = "unknown";
        
        // Анализ паттерна данных для определения типа функции
        private void AnalyzeDataPattern()
        {
            if (xData.Length < 3) return;

            // Безусловно прогоняем синусоидальную подгонку (по x и x²) — даже если в итоге
            // паттерн классифицируется не как trigonometric, GP сможет использовать
            // лучший аналитический seed (если RMSE подгонки достаточно мал).
            TryFitSinusoid(out _);

            // 1. Линейность — самая простая и однозначная проверка
            if (IsLinear())
            {
                dataPattern = "linear";
                return;
            }

            // 2. Тригонометрия раньше полинома: периодичные данные могут случайно
            //    хорошо ложиться на куб (например sin(x) на [0, 2π] ~ кубический).
            int extrema = CountExtrema(yData);
            bool trigLikely = IsTrigonometric();
            if (trigLikely)
            {
                dataPattern = "trigonometric";
                return;
            }

            // 3. Полином: при наличии нескольких экстремумов считаем подозрительным,
            //    но пускаем в полином только если кубический фит близок к идеальному.
            if (IsPolynomial() && extrema < 2)
            {
                dataPattern = "polynomial";
                return;
            }

            // 4. Экспонента
            if (IsExponential())
            {
                dataPattern = "exponential";
                return;
            }

            // 5. Если экстремумов несколько и trig не подошёл — общий профиль
            if (extrema >= 2)
            {
                dataPattern = "general";
                return;
            }

            // 6. Иначе — суперпозиция/общий профиль
            if (IsSuperposition())
            {
                dataPattern = "superposition";
                return;
            }
        }
        
        private bool IsLinear()
        {
            // Проверяем, что разности Y пропорциональны разностям X
            int n = xData.Length;
            if (n < 2) return false;

            double[] diffs = new double[n - 1];
            for (int i = 0; i < n - 1; i++)
            {
                diffs[i] = (yData[i + 1] - yData[i]) / (xData[i + 1] - xData[i]);
            }
            
            double sum = 0;
            for (int i = 0; i < diffs.Length; i++) sum += diffs[i];
            double avgDiff = sum / diffs.Length;

            double varianceSum = 0;
            for (int i = 0; i < diffs.Length; i++)
            {
                double d = diffs[i] - avgDiff;
                varianceSum += d * d;
            }
            double variance = varianceSum / diffs.Length;
            
            return variance < 0.1; // Низкая дисперсия = линейность
        }
        
        private bool IsPolynomial()
        {
            // Проверяем конечные разности для определения степени полинома
            double[] current = yData;
            
            for (int i = 0; i < 3; i++) // Проверяем до 3-й степени
            {
                if (current.Length <= 1) break;
                
                int m = current.Length - 1;
                var next = new double[m];
                for (int j = 0; j < m; j++)
                {
                    next[j] = current[j + 1] - current[j];
                }
                
                // Если разности стали постоянными, это полином
                if (next.Length > 1)
                {
                    double sum = 0;
                    for (int j = 0; j < next.Length; j++) sum += next[j];
                    double avg = sum / next.Length;

                    double varianceSum = 0;
                    for (int j = 0; j < next.Length; j++)
                    {
                        double d = next[j] - avg;
                        varianceSum += d * d;
                    }
                    double variance = varianceSum / next.Length;
                    if (variance < 0.01)
                    {
                        return true;
                    }
                }

                current = next;
            }
            
            // Дополнительная проверка - тестируем полиномиальную регрессию
            return TestPolynomialFit();
        }
        
        // Тестирование полиномиальной регрессии
        private bool TestPolynomialFit()
        {
            double yMin = yData.Min();
            double yMax = yData.Max();
            double yRange = Math.Max(yMax - yMin, 1e-9);

            // Чтобы периодические данные не считались "полиномом" из-за случайно
            // удачной кубической подгонки, требуем, чтобы относительная RMSE
            // была действительно мизерной (<= 1% от размаха).
            for (int degree = 1; degree <= 3; degree++)
            {
                var coefficients = Fit.Polynomial(xData, yData, degree);
                double sse = 0;

                for (int i = 0; i < xData.Length; i++)
                {
                    double x = xData[i];
                    double predicted = 0;
                    double xPow = 1.0;
                    for (int j = 0; j < coefficients.Length; j++)
                    {
                        predicted += coefficients[j] * xPow;
                        xPow *= x;
                    }

                    double diff = yData[i] - predicted;
                    sse += diff * diff;
                }

                double rmse = Math.Sqrt(sse / xData.Length);
                double relRmse = rmse / yRange;
                if (relRmse < 0.01)
                {
                    return true;
                }
            }

            return false;
        }
        
        
        private bool IsExponential()
        {
            // Проверяем, что логарифм Y линейно зависит от X
            // Избегаем LINQ/ToArray в горячей части анализа
            int n = yData.Length;
            if (n < 3) return false;

            // Собираем пары (x, log(y)) только для y>0
            var logY = new List<double>(n);
            var xSubset = new List<double>(n);
            for (int i = 0; i < n; i++)
            {
                double y = yData[i];
                if (y > 0)
                {
                    logY.Add(Math.Log(y));
                    xSubset.Add(xData[i]);
                }
            }
            if (logY.Count < 3) return false;

            int m = logY.Count - 1;
            double[] diffs = new double[m];
            for (int i = 0; i < m; i++)
            {
                diffs[i] = (logY[i + 1] - logY[i]) / (xSubset[i + 1] - xSubset[i]);
            }

            double sum = 0;
            for (int i = 0; i < m; i++) sum += diffs[i];
            double avgDiff = sum / m;

            double varianceSum = 0;
            for (int i = 0; i < m; i++)
            {
                double d = diffs[i] - avgDiff;
                varianceSum += d * d;
            }

            double variance = varianceSum / m;
            return variance < 0.1;
        }
        
        private bool IsTrigonometric()
        {
            int n = yData.Length;
            if (n < 6) return false;

            double minY = yData.Min();
            double maxY = yData.Max();
            double range = maxY - minY;
            if (range < 1e-9 || range > 50) return false;

            int extremaCount = CountExtrema(yData);
            if (extremaCount < 2) return false;

            double xRange = xData[n - 1] - xData[0];
            if (xRange <= 0) return false;

            if (TryFitSinusoid(out double sinusoidRmse) && sinusoidRmse / range < 0.1)
                return true;

            return false;
        }

        private static int CountExtrema(double[] data)
        {
            int count = 0;
            for (int i = 1; i < data.Length - 1; i++)
            {
                if ((data[i] > data[i - 1] && data[i] > data[i + 1]) ||
                    (data[i] < data[i - 1] && data[i] < data[i + 1]))
                {
                    count++;
                }
            }
            return count;
        }

        // Лучшая аналитическая подгонка y ~ a*sin(w*x) + b*cos(w*x) + c0
        // (заполняется TryFitSinusoid и используется как seed для GP).
        private double bestSinA, bestSinB, bestSinC0, bestSinW;
        private bool hasBestSinusoid;
        // bestSinArgKind: какую трансформацию аргумента дала лучшую подгонку
        //   "x"  → стандартная форма a*sin(w*x) + b*cos(w*x) + c0
        //   "x2" → форма с x² внутри: a*sin(w*x²) + b*cos(w*x²) + c0
        // Это нужно, чтобы автоматически распознавать функции вида f(x²)
        // (например, exp(2.1·sin(1.8 + x²)) после log-преобразования).
        private string bestSinArgKind = "x";
        // Относительная RMSE лучшей синусоидальной подгонки (RMSE / yRange).
        // Используется как порог для решения, стоит ли подсевать GP этим seed-ом.
        private double bestSinRelRmse = double.MaxValue;

        // Грубая попытка подогнать y ~ a*sin(w*φ(x)) + b*cos(w*φ(x)) + c
        // на сетке частот w для разных трансформаций φ ∈ { x, x² };
        // возвращает минимальный RMSE по объединённой сетке.
        private bool TryFitSinusoid(out double bestRmse)
        {
            bestRmse = double.MaxValue;
            int n = xData.Length;
            if (n < 4) return false;

            double xMin = xData[0];
            double xMax = xData[n - 1];
            double xRange = xMax - xMin;
            if (xRange <= 0) return false;

            // Подбираем сетки частот в зависимости от аргумента
            // Для аргумента φ(x) сетка частот зависит от размаха φ(x).
            (double[] phi, double phiRange, string kind)[] argVariants;
            {
                double[] phiX = xData;
                double phiXmin = phiX.Min();
                double phiXmax = phiX.Max();
                double phiXrange = phiXmax - phiXmin;

                double[] phiX2 = new double[n];
                for (int i = 0; i < n; i++) phiX2[i] = xData[i] * xData[i];
                double phiX2min = phiX2.Min();
                double phiX2max = phiX2.Max();
                double phiX2range = phiX2max - phiX2min;

                argVariants = new (double[], double, string)[]
                {
                    (phiX, phiXrange, "x"),
                    (phiX2, phiX2range, "x2")
                };
            }

            const int gridSize = 64;
            foreach (var (phi, phiRange, kind) in argVariants)
            {
                if (phiRange <= 0) continue;
                double wMin = 0.5 * Math.PI / phiRange;
                double wMax = Math.PI * (n - 1) / phiRange;
                if (wMax <= wMin) continue;

                for (int g = 0; g < gridSize; g++)
                {
                    double w = wMin + (wMax - wMin) * g / (gridSize - 1);

                    double sumS = 0, sumC = 0, sumSS = 0, sumCC = 0, sumSC = 0;
                    double sumYS = 0, sumYC = 0, sumY = 0;
                    for (int i = 0; i < n; i++)
                    {
                        double s = Math.Sin(w * phi[i]);
                        double c = Math.Cos(w * phi[i]);
                        double y = yData[i];
                        sumS += s;
                        sumC += c;
                        sumSS += s * s;
                        sumCC += c * c;
                        sumSC += s * c;
                        sumYS += y * s;
                        sumYC += y * c;
                        sumY += y;
                    }

                    double m11 = sumSS, m12 = sumSC, m13 = sumS;
                    double m21 = sumSC, m22 = sumCC, m23 = sumC;
                    double m31 = sumS, m32 = sumC, m33 = n;
                    double det = Det3(m11, m12, m13, m21, m22, m23, m31, m32, m33);
                    if (Math.Abs(det) < 1e-12) continue;

                    double a = Det3(sumYS, m12, m13, sumYC, m22, m23, sumY, m32, m33) / det;
                    double b = Det3(m11, sumYS, m13, m21, sumYC, m23, m31, sumY, m33) / det;
                    double c0 = Det3(m11, m12, sumYS, m21, m22, sumYC, m31, m32, sumY) / det;

                    double sse = 0;
                    for (int i = 0; i < n; i++)
                    {
                        double pred = a * Math.Sin(w * phi[i]) + b * Math.Cos(w * phi[i]) + c0;
                        double diff = yData[i] - pred;
                        sse += diff * diff;
                    }
                    double rmse = Math.Sqrt(sse / n);
                    if (rmse < bestRmse)
                    {
                        bestRmse = rmse;
                        bestSinA = a;
                        bestSinB = b;
                        bestSinC0 = c0;
                        bestSinW = w;
                        bestSinArgKind = kind;
                        hasBestSinusoid = true;
                    }
                }
            }

            // Сохраняем относительный RMSE для решения о сидинге GP
            double yRange = yData.Max() - yData.Min();
            bestSinRelRmse = yRange > 0 ? bestRmse / yRange : double.MaxValue;

            return bestRmse < double.MaxValue;
        }

        // Возвращает дерево вида (a*sin(w*φ(x1)) + b*cos(w*φ(x1))) + c0,
        // где φ — либо x, либо x² (выбирается по результатам аналитической подгонки).
        // Если подгонки нет — null.
        private ExpressionNode? CreateSinusoidSeed()
        {
            if (!hasBestSinusoid) return null;

            ExpressionNode VarX() => new ExpressionNode { Operation = "var", VariableIndex = 0 };
            ExpressionNode Const(double v) => new ExpressionNode { Operation = "const", Constant = v };

            ExpressionNode arg = bestSinArgKind == "x2"
                ? new ExpressionNode { Operation = "*", Children = { VarX(), VarX() } }
                : VarX();
            var wx = new ExpressionNode { Operation = "*", Children = { Const(bestSinW), arg } };
            var sinPart = new ExpressionNode
            {
                Operation = "*",
                Children =
                {
                    Const(bestSinA),
                    new ExpressionNode { Operation = "sin", Children = { wx } }
                }
            };
            var cosPart = new ExpressionNode
            {
                Operation = "*",
                Children =
                {
                    Const(bestSinB),
                    new ExpressionNode { Operation = "cos", Children = { wx.Clone() } }
                }
            };
            var sinPlusCos = new ExpressionNode
            {
                Operation = "+",
                Children = { sinPart, cosPart }
            };
            var withOffset = new ExpressionNode
            {
                Operation = "+",
                Children = { sinPlusCos, Const(bestSinC0) }
            };
            return withOffset;
        }

        private static double Det3(
            double a11, double a12, double a13,
            double a21, double a22, double a23,
            double a31, double a32, double a33)
        {
            return a11 * (a22 * a33 - a23 * a32)
                 - a12 * (a21 * a33 - a23 * a31)
                 + a13 * (a21 * a32 - a22 * a31);
        }
        
        private bool IsSuperposition()
        {
            // Проверяем сложные паттерны, которые не подходят под другие категории
            return !IsLinear() && !IsPolynomial() && !IsExponential() && !IsTrigonometric();
        }
        
        // Метод для получения типа данных
        public string GetDataPattern()
        {
            return dataPattern;
        }

        // Создание простого полинома

        // Создание случайного дерева
        private ExpressionNode CreateRandomTree(int depth = 0)
        {
            var node = new ExpressionNode { Depth = depth };
            
            if (depth >= maxDepth)
            {
                // Листовой узел
                // Терминалы для n-мерного режима:
                // - const
                // - var(xi) или base(fi) в режиме суперпозиции
                if (random.NextDouble() < config.TerminalConstProbability)
                {
                    node.Operation = "const";
                    node.Constant = random.NextDouble() * 10 - 5;
                    return node;
                }

                if (superpositionMode)
                {
                    node.Operation = "base";
                    node.BaseFunctionIndex = random.Next(baseFunctions.Count);
                    return node;
                }

                node.Operation = "var";
                node.VariableIndex = variableCount > 0 ? random.Next(variableCount) : 0;
                return node;
            }

            // Увеличиваем вероятность создания сложных выражений, но очень редко
            if (depth < 2 && random.NextDouble() < 0.1)
            {
                // Принудительно создаем сложные выражения на верхних уровнях
                var complexOperations = config.AllowedOps.Where(o => o is "sin" or "cos" or "exp" or "log" or "pow" or "neg").ToArray();
                if (complexOperations.Length == 0)
                    complexOperations = config.AllowedOps;
                var complexOperation = complexOperations[random.Next(complexOperations.Length)];
                node.Operation = complexOperation;

                switch (complexOperation)
                {
                    case "sin":
                    case "cos":
                    case "exp":
                    case "log":
                    case "neg":
                        node.Children.Add(CreateRandomTree(depth + 1));
                        break;
                    case "pow":
                        node.Children.Add(CreateRandomTree(depth + 1));
                        var powerNode = new ExpressionNode { Operation = "const", Constant = random.Next(1, 4), Depth = depth + 1 };
                        node.Children.Add(powerNode);
                        break;
                }
                return node;
            }

            // Выбор операции из профиля грамматики
            string operation = config.AllowedOps[random.Next(config.AllowedOps.Length)];
            node.Operation = operation;

            switch (operation)
            {
                case "+":
                case "-":
                case "*":
                case "/":
                    node.Children.Add(CreateRandomTree(depth + 1));
                    node.Children.Add(CreateRandomTree(depth + 1));
                    break;
                case "sin":
                case "cos":
                case "exp":
                case "log":
                case "neg":
                    node.Children.Add(CreateRandomTree(depth + 1));
                    break;
                case "pow":
                    node.Children.Add(CreateRandomTree(depth + 1));
                    // Второй аргумент - степень (обычно небольшая)
                    var powerNode = new ExpressionNode { Operation = "const", Constant = random.Next(1, 4), Depth = depth + 1 };
                    node.Children.Add(powerNode);
                    break;
            }

            return node;
        }

        // Вычисление значения выражения
        private double Evaluate(ExpressionNode node, double x)
        {
            try
            {
                // legacy 1D wrapper
                return Evaluate(node, new[] { x });
            }
            catch
            {
                return double.NaN;
            }
        }
        
        // Вычисление значения выражения для множественных переменных
        private double Evaluate(ExpressionNode node, double x1, double x2)
        {
            try
            {
                // legacy 2D wrapper
                return Evaluate(node, new[] { x1, x2 });
            }
            catch
            {
                return double.NaN;
            }
        }

        // Универсальное вычисление дерева для n переменных (vars[0]=x1,...)
        private double Evaluate(ExpressionNode node, double[] vars)
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
                    int idx = node.BaseFunctionIndex ?? -1;
                    if (idx < 0 || idx >= baseFunctions.Count) return double.NaN;
                    return Evaluate(baseFunctions[idx], vars);
                }
                case "neg":
                {
                    if (node.Children.Count < 1) return double.NaN;
                    double v = Evaluate(node.Children[0], vars);
                    return (double.IsNaN(v) || double.IsInfinity(v)) ? double.NaN : -v;
                }
                case "+":
                case "-":
                case "*":
                case "/":
                {
                    if (node.Children.Count < 2) return double.NaN;
                    double a = Evaluate(node.Children[0], vars);
                    double b = Evaluate(node.Children[1], vars);
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
                    double a = Evaluate(node.Children[0], vars);
                    if (double.IsNaN(a) || double.IsInfinity(a)) return double.NaN;
                    return node.Operation switch
                    {
                        "sin" => Math.Sin(a),
                        "cos" => Math.Cos(a),
                        "exp" => a > config.MaxExpArg ? double.PositiveInfinity : Math.Exp(a),
                        "log" => a <= 0 ? double.NaN : Math.Log(a),
                        _ => double.NaN
                    };
                }
                case "pow":
                {
                    if (node.Children.Count < 2) return double.NaN;
                    double a = Evaluate(node.Children[0], vars);
                    double p = Evaluate(node.Children[1], vars);
                    if (double.IsNaN(a) || double.IsNaN(p) || double.IsInfinity(a) || double.IsInfinity(p))
                        return double.NaN;
                    if (Math.Abs(p) > config.MaxPowAbs) return double.NaN;
                    if (a < 0 && p != Math.Floor(p)) return double.NaN;
                    return Math.Pow(a, p);
                }
                default:
                    return double.NaN;
            }
        }

        // Вычисление приспособленности (fitness)
        private double CalculateFitness(ExpressionNode tree)
        {
            int n = yData.Length;
            if (n == 0) return double.MaxValue;

            double[][] evalX = BuildCurrentDatasetX();

            double sse = 0;
            int validPoints = 0;
            int invalidPoints = 0;

            for (int i = 0; i < n; i++)
            {
                double predicted = Evaluate(tree, evalX[i]);
                if (double.IsNaN(predicted) || double.IsInfinity(predicted))
                {
                    invalidPoints++;
                    continue;
                }

                double diff = yData[i] - predicted;
                sse += diff * diff;
                validPoints++;
            }

            if (validPoints == 0)
                return double.MaxValue / 2;

            double mse = Math.Max(sse / validPoints, 1e-12);
            int complexity = CountNodes(tree);
            double invalidRatio = (double)invalidPoints / n;

            // Универсальная устойчивая fitness-функция:
            // ошибка + сложность + штраф за невалидные предсказания.
            return mse
                   + config.ComplexityPenalty * complexity
                   + config.InvalidPredictionPenalty * invalidRatio;
        }

        // Подсчет узлов в дереве
        private int CountNodes(ExpressionNode node)
        {
            int count = 1;
            foreach (var child in node.Children)
            {
                count += CountNodes(child);
            }
            return count;
        }

        /// <summary>Высота дерева (лист = 1). Не опирается на устаревшее поле <see cref="ExpressionNode.Depth"/>.</summary>
        private static int GetMaxDepth(ExpressionNode node)
        {
            if (node.Children.Count == 0) return 1;
            int m = 0;
            foreach (var ch in node.Children)
                m = Math.Max(m, GetMaxDepth(ch));
            return 1 + m;
        }

        private static void SetDepthRecursive(ExpressionNode node, int depth)
        {
            node.Depth = depth;
            foreach (var ch in node.Children)
                SetDepthRecursive(ch, depth + 1);
        }

        /// <summary>Лист для shrink-мутации (этап D).</summary>
        private ExpressionNode CreateRandomTerminal(int depth)
        {
            var node = new ExpressionNode { Depth = depth };
            if (random.NextDouble() < config.TerminalConstProbability)
            {
                node.Operation = "const";
                node.Constant = random.NextDouble() * 10 - 5;
                return node;
            }

            if (superpositionMode && baseFunctions.Count > 0)
            {
                node.Operation = "base";
                node.BaseFunctionIndex = random.Next(baseFunctions.Count);
                return node;
            }

            node.Operation = "var";
            node.VariableIndex = variableCount > 0 ? random.Next(variableCount) : 0;
            return node;
        }

        /// <summary>Если кроссовер/мутация дали превышение <see cref="maxDepth"/> — схлопываем случайные внутренние узлы.</summary>
        private void ClampTreeDepth(ExpressionNode root)
        {
            for (int guard = 0; guard < 100 && GetMaxDepth(root) > maxDepth; guard++)
            {
                var nodes = GetAllNodes(root);
                var internalNodes = new List<ExpressionNode>();
                foreach (var n in nodes)
                {
                    if (n.Children.Count > 0)
                        internalNodes.Add(n);
                }
                if (internalNodes.Count == 0)
                    break;
                var victim = internalNodes[random.Next(internalNodes.Count)];
                ReplaceSubtree(victim, CreateRandomTerminal(victim.Depth));
            }
        }

        // Мутация (этап D: hoist / shrink / сильнее константы / subtree)
        private void Mutate(ExpressionNode node)
        {
            MaybeMutateThisNode(node);
            foreach (var child in node.Children)
                Mutate(child);
        }

        private void MaybeMutateThisNode(ExpressionNode node)
        {
            if (random.NextDouble() >= mutationRate)
                return;

            if (node.Operation == "const")
            {
                if (random.NextDouble() < 0.78)
                    node.Constant = (node.Constant ?? 0) + (random.NextDouble() - 0.5) * 0.5;
                else
                    node.Constant = (node.Constant ?? 0) + (random.NextDouble() - 0.5) * 5.0;
                return;
            }

            double r = random.NextDouble();
            if (r < 0.18 && node.Children.Count > 0)
            {
                var donor = node.Children[random.Next(node.Children.Count)];
                ReplaceSubtree(node, donor.Clone());
                return;
            }
            if (r < 0.36)
            {
                ReplaceSubtree(node, CreateRandomTree(node.Depth));
                return;
            }
            if (r < 0.48)
            {
                ReplaceSubtree(node, CreateRandomTerminal(node.Depth));
                return;
            }

            node.Operation = config.AllowedOps[random.Next(config.AllowedOps.Length)];

            if (node.Operation is "sin" or "cos" or "exp" or "log" or "neg")
            {
                while (node.Children.Count < 1)
                    node.Children.Add(CreateRandomTree(node.Depth + 1));
                while (node.Children.Count > 1)
                    node.Children.RemoveAt(node.Children.Count - 1);
            }
            else
            {
                while (node.Children.Count < 2)
                    node.Children.Add(CreateRandomTree(node.Depth + 1));
            }

            if (node.Operation == "pow")
            {
                while (node.Children.Count < 2)
                    node.Children.Add(CreateRandomTree(node.Depth + 1));
                node.Children[1].Operation = "const";
                node.Children[1].Constant = random.Next(1, 4);
                node.Children[1].VariableIndex = null;
                node.Children[1].BaseFunctionIndex = null;
                node.Children[1].Children.Clear();
            }
        }

        // Скрещивание (этап D: повторные попытки, лимит глубины)
        private ExpressionNode Crossover(ExpressionNode parent1, ExpressionNode parent2)
        {
            if (random.NextDouble() > crossoverRate)
                return parent1.Clone();

            const int maxAttempts = 10;
            for (int attempt = 0; attempt < maxAttempts; attempt++)
            {
                var result = parent1.Clone();
                var nodes1 = GetAllNodes(result);
                var nodes2 = GetAllNodes(parent2);
                if (nodes1.Count > 0 && nodes2.Count > 0)
                {
                    var randomNode1 = nodes1[random.Next(nodes1.Count)];
                    var randomNode2 = nodes2[random.Next(nodes2.Count)];
                    ReplaceSubtree(randomNode1, randomNode2.Clone());
                }
                if (GetMaxDepth(result) <= maxDepth)
                    return result;
            }

            return parent1.Clone();
        }

        // Получение всех узлов дерева
        private List<ExpressionNode> GetAllNodes(ExpressionNode node)
        {
            var nodes = new List<ExpressionNode> { node };
            foreach (var child in node.Children)
            {
                nodes.AddRange(GetAllNodes(child));
            }
            return nodes;
        }

        // Замена поддерева
        private void ReplaceSubtree(ExpressionNode target, ExpressionNode replacement)
        {
            target.Operation = replacement.Operation;
            target.Constant = replacement.Constant;
            target.VariableIndex = replacement.VariableIndex;
            target.BaseFunctionIndex = replacement.BaseFunctionIndex;
            target.Children.Clear();
            foreach (var child in replacement.Children)
            {
                target.Children.Add(child.Clone());
            }
        }

        /// <summary>
        /// Главный цикл GP.
        /// </summary>
        /// <remarks>
        /// Внутри одного вызова входные данные (уже те, что переданы в движок) один раз поделены на
        /// внутренний train и validation (~80/20, см. <see cref="SplitTrainValidation"/>).
        /// Эволюция идёт на внутреннем train; лучший среди рестартов выбирается по RMSE на этой внутренней validation.
        /// Это уровень «механики отбора дерева» и не подменяет внешний test в multi-var (там — отдельное разбиение по gp_settings).
        /// </remarks>
        public (ExpressionNode bestTree, double bestFitness) Evolve()
        {
            int sessionBase = GpRandomizeBaseEachRun ? Random.Shared.Next() : GpRandomBaseSeed;
            int splitSeed = GpRandomizeBaseEachRun ? (sessionBase ^ unchecked((int)0x9E3779B9)) : 42;
            var (trainX, trainY, valX, valY) = SplitTrainValidation(0.2, splitSeed);

            ExpressionNode? bestTree = null;
            double bestValRmse = double.MaxValue;
            double bestFitness = double.MaxValue;

            // Мульти-старт: каждый рестарт эволюционирует только на внутреннем train; сравнение рестартов — по RMSE на внутренней validation
            for (int restart = 0; restart < config.Restarts; restart++)
            {
                random = new Random(sessionBase + restart);
                var (candidate, trainFitness) = RunSingleEvolution(trainX, trainY, restart);

                // Локальная подстройка констант (на внутреннем train); выбор кандидата — по RMSE на внутренней validation
                var optimized = OptimizeConstants(candidate, trainX, trainY);
                var optimizedTrainFitness = CalculateFitnessOnDataset(optimized, trainX, trainY);
                var candidateValRmse = CalculateRmse(optimized, valX, valY);

                if (candidateValRmse < bestValRmse || (Math.Abs(candidateValRmse - bestValRmse) < 1e-12 && optimizedTrainFitness < bestFitness))
                {
                    bestValRmse = candidateValRmse;
                    bestFitness = optimizedTrainFitness;
                    bestTree = optimized.Clone();
                }
            }

            if (bestTree == null)
                bestTree = CreateRandomTree();

            // train fitness выбранного по внутренней validation кандидата (см. GpTrainObjectiveReportLabel)
            return (bestTree, bestFitness);
        }

        /// <summary>
        /// Вложенное разбиение внутри текущей задачи GP: внутренний train / внутренняя validation для эволюции и отбора рестарта.
        /// Не является «тестом для сравнения моделей» с baseline; для multi-var внешняя оценка задаётся отдельно (multiVarValidation).
        /// </summary>
        private (double[][] trainX, double[] trainY, double[][] valX, double[] valY) SplitTrainValidation(double validationPart, int seed)
        {
            var X = BuildCurrentDatasetX();
            int n = yData.Length;
            int valCount = Math.Max(1, (int)Math.Round(n * validationPart));
            int trainCount = Math.Max(1, n - valCount);
            if (trainCount + valCount > n) valCount = n - trainCount;

            var idx = Enumerable.Range(0, n).ToArray();
            var rng = new Random(seed);
            for (int i = n - 1; i > 0; i--)
            {
                int j = rng.Next(i + 1);
                (idx[i], idx[j]) = (idx[j], idx[i]);
            }

            var trainX = new double[trainCount][];
            var trainY = new double[trainCount];
            var valX = new double[valCount][];
            var valY = new double[valCount];

            for (int i = 0; i < trainCount; i++)
            {
                trainX[i] = X[idx[i]];
                trainY[i] = yData[idx[i]];
            }
            for (int i = 0; i < valCount; i++)
            {
                valX[i] = X[idx[trainCount + i]];
                valY[i] = yData[idx[trainCount + i]];
            }

            return (trainX, trainY, valX, valY);
        }

        private static void AppendEvolutionTrace(string text)
        {
            var tr = GlobalEvolutionTrace;
            if (tr?.Enabled != true || string.IsNullOrWhiteSpace(tr.LogFile)) return;
            try
            {
                string path = Path.IsPathRooted(tr.LogFile)
                    ? tr.LogFile
                    : Path.Combine(Environment.CurrentDirectory, tr.LogFile);
                lock (EvolutionTraceLock)
                    File.AppendAllText(path, text, Encoding.UTF8);
            }
            catch
            {
                /* игнорируем ошибки логирования */
            }
        }

        private (ExpressionNode bestTree, double bestFitness) RunSingleEvolution(double[][] trainX, double[] trainY, int restartIndex)
        {
            var population = new List<ExpressionNode>(populationSize);

            // Подсев аналитической подгонкой sin/cos: даём GP сильную стартовую гипотезу
            // вида (a*sin(w*φ(x)) + b*cos(w*φ(x))) + c, где φ ∈ {x, x²}.
            // Это резко ускоряет поиск формул вроде sin(x), 2·sin(3x+1), sin(x²+c) и т.п.
            // Сидим, если относительная RMSE подгонки < 30% — иначе seed бессмысленный.
            // Универсальное условие (не только trigonometric): помогает и в log-fit,
            // где паттерн помечен как general, но структура всё равно может быть
            // тригонометрической (как при exp(sin(x²+c))).
            if (hasBestSinusoid && bestSinRelRmse < 0.3)
            {
                var seed = CreateSinusoidSeed();
                if (seed != null)
                {
                    int seedClones = Math.Min(populationSize / 10, 20);
                    for (int i = 0; i < seedClones; i++)
                        population.Add(seed.Clone());
                }
            }

            while (population.Count < populationSize)
                population.Add(CreateRandomTree(0));

            ExpressionNode bestTree = population[0].Clone();
            double bestFitness = double.MaxValue;

            for (int generation = 0; generation < generations; generation++)
            {
                var scored = new List<(ExpressionNode tree, double fitness)>(population.Count);
                foreach (var tree in population)
                {
                    var fit = CalculateFitnessOnDataset(tree, trainX, trainY);
                    scored.Add((tree, fit));
                    if (fit < bestFitness)
                    {
                        bestFitness = fit;
                        bestTree = tree.Clone();
                    }
                }

                scored.Sort((a, b) => a.fitness.CompareTo(b.fitness));

                var trCfg = GlobalEvolutionTrace;
                if (trCfg?.Enabled == true)
                {
                    int every = Math.Max(1, trCfg.EveryGenerations);
                    if (generation % every == 0 || generation == generations - 1)
                    {
                        int top = Math.Min(Math.Max(1, trCfg.TopK), scored.Count);
                        var sb = new StringBuilder();
                        sb.Append('[').Append(EvolutionTraceRunLabel).Append("] restart=").Append(restartIndex)
                            .Append(" gen=").Append(generation + 1).Append('/').Append(generations).Append('\n');
                        for (int i = 0; i < top; i++)
                        {
                            double trRmse = CalculateRmse(scored[i].tree, trainX, trainY);
                            sb.Append("  #").Append(i + 1).Append(" fitness=").Append(scored[i].fitness.ToString("F8"))
                                .Append(" trainRMSE=").Append(trRmse.ToString("F8")).Append("  ").Append(scored[i].tree)
                                .Append('\n');
                        }

                        sb.Append('\n');
                        AppendEvolutionTrace(sb.ToString());
                    }
                }

                var next = new List<ExpressionNode>(populationSize)
                {
                    scored[0].tree.Clone()
                };
                if (scored.Count > 1) next.Add(scored[1].tree.Clone());

                while (next.Count < populationSize)
                {
                    var parent1 = TournamentSelection(scored);
                    var parent2 = TournamentSelection(scored);
                    var child = Crossover(parent1, parent2);
                    Mutate(child);
                    SetDepthRecursive(child, 0);
                    ClampTreeDepth(child);
                    SetDepthRecursive(child, 0);
                    next.Add(child);
                }

                population = next;
            }

            return (bestTree, bestFitness);
        }

        /// <summary>Текущие входы для Evaluate: либо полная матрица X (multi-var), либо один столбец x (1-var).</summary>
        private double[][] BuildCurrentDatasetX()
        {
            if (XData != null)
                return XData;

            int n = yData.Length;
            var single = new double[n][];
            for (int i = 0; i < n; i++)
                single[i] = new[] { xData[i] };
            return single;
        }

        private double CalculateFitnessOnDataset(ExpressionNode tree, double[][] x, double[] y)
        {
            double sse = 0;
            int valid = 0;
            int invalid = 0;
            for (int i = 0; i < y.Length; i++)
            {
                double predicted = Evaluate(tree, x[i]);
                if (double.IsNaN(predicted) || double.IsInfinity(predicted))
                {
                    invalid++;
                    continue;
                }

                double diff = y[i] - predicted;
                sse += diff * diff;
                valid++;
            }

            if (valid == 0) return double.MaxValue / 2;
            double mse = Math.Max(sse / valid, 1e-12);
            double invalidRatio = (double)invalid / y.Length;
            return mse + config.ComplexityPenalty * CountNodes(tree) + config.InvalidPredictionPenalty * invalidRatio;
        }

        private double CalculateRmse(ExpressionNode tree, double[][] x, double[] y)
        {
            double sse = 0;
            int valid = 0;
            for (int i = 0; i < y.Length; i++)
            {
                double predicted = Evaluate(tree, x[i]);
                if (double.IsNaN(predicted) || double.IsInfinity(predicted))
                    continue;

                double diff = y[i] - predicted;
                sse += diff * diff;
                valid++;
            }

            if (valid == 0) return double.MaxValue / 2;
            return Math.Sqrt(sse / valid);
        }

        private ExpressionNode OptimizeConstants(ExpressionNode sourceTree, double[][] trainX, double[] trainY)
        {
            var tree = sourceTree.Clone();
            var constants = new List<ExpressionNode>();
            CollectConstantNodes(tree, constants);
            if (constants.Count == 0) return tree;

            CoordinateConstantRefinement(tree, constants, trainX, trainY);

            if (constants.Count <= 24)
            {
                try
                {
                    var initial = Vector<double>.Build.Dense(constants.Count);
                    var perturb = Vector<double>.Build.Dense(constants.Count);
                    for (int i = 0; i < constants.Count; i++)
                    {
                        double c = constants[i].Constant ?? 0;
                        initial[i] = c;
                        double ac = Math.Abs(c);
                        perturb[i] = ac < 1e-9 ? 0.08 : Math.Min(0.35 * ac + 0.05, 3.0);
                    }

                    var obj = new TreeConstantObjective(this, tree, constants, trainX, trainY);
                    int maxIter = Math.Min(6000, 400 + 250 * constants.Count);
                    var nm = new NelderMeadSimplex(1e-7, maxIter);
                    nm.FindMinimum(obj, initial, perturb);
                }
                catch (MaximumIterationsException)
                {
                    CoordinateConstantRefinement(tree, constants, trainX, trainY);
                }
            }

            return tree;
        }

        /// <summary>Покоординатный спуск по константам — быстрый baseline и подстраховка после Nelder–Mead.</summary>
        private void CoordinateConstantRefinement(ExpressionNode tree, List<ExpressionNode> constants, double[][] trainX, double[] trainY)
        {
            for (int pass = 0; pass < 3; pass++)
            {
                foreach (var c in constants)
                {
                    if (!c.Constant.HasValue) continue;
                    double current = c.Constant.Value;
                    double best = CalculateFitnessOnDataset(tree, trainX, trainY);
                    double step = Math.Max(0.02, Math.Min(3.0, Math.Abs(current) * 0.2 + 0.12));

                    for (int it = 0; it < 22; it++)
                    {
                        c.Constant = current + step;
                        double plus = CalculateFitnessOnDataset(tree, trainX, trainY);
                        c.Constant = current - step;
                        double minus = CalculateFitnessOnDataset(tree, trainX, trainY);
                        c.Constant = current;

                        if (plus < best && plus <= minus)
                        {
                            current += step;
                            best = plus;
                        }
                        else if (minus < best)
                        {
                            current -= step;
                            best = minus;
                        }
                        else
                        {
                            step *= 0.55;
                        }

                        c.Constant = current;
                    }
                }
            }
        }

        /// <summary>Целевая функция для Math.NET: MSE+штрафы по узлам <c>const</c> в клоне дерева.</summary>
        private sealed class TreeConstantObjective : IObjectiveFunction
        {
            private readonly SymbolicRegressionEngine _eng;
            private readonly ExpressionNode _tree;
            private readonly List<ExpressionNode> _constNodes;
            private readonly double[][] _trainX;
            private readonly double[] _trainY;
            private Vector<double> _point;
            private double _value;

            public TreeConstantObjective(
                SymbolicRegressionEngine eng,
                ExpressionNode tree,
                List<ExpressionNode> constNodes,
                double[][] trainX,
                double[] trainY)
            {
                _eng = eng;
                _tree = tree;
                _constNodes = constNodes;
                _trainX = trainX;
                _trainY = trainY;
                _point = Vector<double>.Build.Dense(constNodes.Count);
                for (int i = 0; i < constNodes.Count; i++)
                    _point[i] = constNodes[i].Constant ?? 0;
                EvaluateAt(_point);
            }

            public Vector<double> Point => _point;
            public double Value => _value;
            public bool IsGradientSupported => false;
            public Vector<double> Gradient => throw new NotSupportedException();
            public bool IsHessianSupported => false;
            public MathNet.Numerics.LinearAlgebra.Matrix<double> Hessian => throw new NotSupportedException();

            public void EvaluateAt(Vector<double> point)
            {
                _point = point;
                for (int i = 0; i < _constNodes.Count; i++)
                    _constNodes[i].Constant = point[i];
                _value = _eng.CalculateFitnessOnDataset(_tree, _trainX, _trainY);
            }

            public IObjectiveFunction Fork()
            {
                var t = _tree.Clone();
                var list = new List<ExpressionNode>();
                _eng.CollectConstantNodes(t, list);
                var o = new TreeConstantObjective(_eng, t, list, _trainX, _trainY);
                o.EvaluateAt(_point);
                return o;
            }

            public IObjectiveFunction CreateNew() => Fork();
        }

        private void CollectConstantNodes(ExpressionNode node, List<ExpressionNode> constants)
        {
            if (node.Operation == "const")
                constants.Add(node);

            foreach (var child in node.Children)
                CollectConstantNodes(child, constants);
        }
        
        private ExpressionNode TournamentSelection(List<(ExpressionNode tree, double fitness)> fitness)
        {
            int tournamentSize = 3;
            var tournament = new List<(ExpressionNode tree, double fitness)>();
            
            for (int i = 0; i < tournamentSize; i++)
            {
                int index = random.Next(fitness.Count);
                tournament.Add(fitness[index]);
            }
            
            return tournament.OrderBy(t => t.fitness).First().tree;
        }
    }
}
