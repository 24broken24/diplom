using System;
using System.Collections.Generic;
using System.IO;
using System.Text.Json;
using System.Text.Json.Serialization;

namespace SymbolicRegression
{
    public sealed class GpProfileOverride
    {
        public int? MaxDepth { get; set; }
        public int? PopulationSize { get; set; }
        public int? Generations { get; set; }
        public int? Restarts { get; set; }
        public double? MutationRate { get; set; }
        public double? CrossoverRate { get; set; }
        public double? TerminalConstProbability { get; set; }
        public double? ComplexityPenalty { get; set; }
        public double? InvalidPredictionPenalty { get; set; }
        public double? MaxExpArg { get; set; }
        public double? MaxPowAbs { get; set; }
        public string[]? AllowedOps { get; set; }
    }

    public sealed class AppRuntimeConfig
    {
        [JsonPropertyName("gpProfiles")]
        public Dictionary<string, GpProfileOverride> GpProfiles { get; set; } = new(StringComparer.OrdinalIgnoreCase);

        [JsonPropertyName("evaluationMaxRelativeRmse")]
        public double? EvaluationMaxRelativeRmse { get; set; }

        [JsonPropertyName("evaluationMaxAbsRmse")]
        public double? EvaluationMaxAbsRmse { get; set; }

        [JsonPropertyName("gpEvolutionTrace")]
        public GpEvolutionTraceConfig? GpEvolutionTrace { get; set; }

        [JsonPropertyName("gpRandomBaseSeed")]
        public int? GpRandomBaseSeed { get; set; }

        [JsonPropertyName("gpRandomizeBaseEachRun")]
        public bool GpRandomizeBaseEachRun { get; set; }

        [JsonPropertyName("includeOneVarLogFit")]
        public bool? IncludeOneVarLogFit { get; set; }

        [JsonPropertyName("includeClassicRegressionBaselines")]
        public bool? IncludeClassicRegressionBaselines { get; set; }

        [JsonPropertyName("multiVarValidation")]
        public MultiVarValidationConfig? MultiVarValidation { get; set; }
    }

    public sealed class MultiVarValidationConfig
    {
        [JsonPropertyName("repeats")]
        public int? Repeats { get; set; }

        [JsonPropertyName("testFraction")]
        public double? TestFraction { get; set; }

        [JsonPropertyName("splitMode")]
        public string? SplitMode { get; set; }

        [JsonPropertyName("kFolds")]
        public int? KFolds { get; set; }

        [JsonPropertyName("shuffleSeed")]
        public int? ShuffleSeed { get; set; }
    }


    public sealed class GpEvolutionTraceConfig
    {
        [JsonPropertyName("enabled")]
        public bool Enabled { get; set; }

        [JsonPropertyName("everyGenerations")]
        public int EveryGenerations { get; set; } = 5;

        [JsonPropertyName("topK")]
        public int TopK { get; set; } = 5;

        [JsonPropertyName("logFile")]
        public string LogFile { get; set; } = "gp_evolution_trace.txt";
    }

    public static class RuntimeConfigLoader
    {
        public static AppRuntimeConfig Load()
        {
            var empty = new AppRuntimeConfig
            {
                GpProfiles = new Dictionary<string, GpProfileOverride>(StringComparer.OrdinalIgnoreCase)
            };

            try
            {
                var configPath = Path.Combine(Environment.CurrentDirectory, "gp_settings.json");
                if (!File.Exists(configPath)) return empty;

                var json = File.ReadAllText(configPath);
                var loaded = JsonSerializer.Deserialize<AppRuntimeConfig>(json, new JsonSerializerOptions
                {
                    ReadCommentHandling = JsonCommentHandling.Skip,
                    PropertyNameCaseInsensitive = true
                });

                if (loaded == null) return empty;

                if (loaded.GpProfiles == null)
                    loaded.GpProfiles = new Dictionary<string, GpProfileOverride>(StringComparer.OrdinalIgnoreCase);

                return loaded;
            }
            catch
            {
                return empty;
            }
        }
    }
}
