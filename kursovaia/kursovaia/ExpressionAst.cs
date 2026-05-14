using System;
using System.Collections.Generic;
using System.Globalization;
using System.Linq;
using System.Text.RegularExpressions;

namespace SymbolicRegression
{
    internal sealed class ExpressionParser
    {
        private readonly string _s;
        private int _i;
        private readonly int _varCount;

        public ExpressionParser(string s, int varCount)
        {
            _s = s;
            _varCount = varCount;
        }

        public ExpressionNode Parse()
        {
            _i = 0;
            var expr = ParseAddSub();
            SkipWs();
            if (_i < _s.Length)
                throw new FormatException($"Unexpected token at pos {_i}: '{_s[_i]}'");
            return expr;
        }

        private ExpressionNode ParseAddSub()
        {
            var left = ParseMulDiv();
            while (true)
            {
                SkipWs();
                if (TryConsume('+'))
                {
                    var right = ParseMulDiv();
                    left = new ExpressionNode { Operation = "+", Children = { left, right } };
                    continue;
                }
                if (TryConsume('-'))
                {
                    var right = ParseMulDiv();
                    left = new ExpressionNode { Operation = "-", Children = { left, right } };
                    continue;
                }
                return left;
            }
        }

        private ExpressionNode ParseMulDiv()
        {
            var left = ParsePow();
            while (true)
            {
                SkipWs();
                if (TryConsume('*'))
                {
                    var right = ParsePow();
                    left = new ExpressionNode { Operation = "*", Children = { left, right } };
                    continue;
                }
                if (TryConsume('/'))
                {
                    var right = ParsePow();
                    left = new ExpressionNode { Operation = "/", Children = { left, right } };
                    continue;
                }
                return left;
            }
        }

        private ExpressionNode ParsePow()
        {
            var left = ParseUnary();
            SkipWs();
            if (TryConsume('^'))
            {
                var right = ParsePow();
                return new ExpressionNode { Operation = "pow", Children = { left, right } };
            }
            return left;
        }

        private ExpressionNode ParseUnary()
        {
            SkipWs();
            if (TryConsume('-'))
            {
                var inner = ParseUnary();
                return new ExpressionNode { Operation = "neg", Children = { inner } };
            }

            // func(...)
            string? ident = TryReadIdentifier();
            if (ident != null)
            {
                if (ident.StartsWith("x", StringComparison.OrdinalIgnoreCase))
                {
                    if (!int.TryParse(ident.AsSpan(1), out int num) || num < 1 || num > _varCount)
                        throw new FormatException($"Unknown variable '{ident}'");
                    return new ExpressionNode { Operation = "var", VariableIndex = num - 1 };
                }

                if (ident is "sin" or "cos" or "exp" or "log")
                {
                    SkipWs();
                    Expect('(');
                    var arg = ParseAddSub();
                    SkipWs();
                    Expect(')');
                    return new ExpressionNode { Operation = ident, Children = { arg } };
                }

                // формат для пи 
                throw new FormatException($"Unknown identifier '{ident}'");
            }

            SkipWs();
            if (TryConsume('('))
            {
                var inner = ParseAddSub();
                SkipWs();
                Expect(')');
                return inner;
            }

            return ParseNumber();
        }

        private ExpressionNode ParseNumber()
        {
            SkipWs();
            int start = _i;
            while (_i < _s.Length)
            {
                char c = _s[_i];
                if (char.IsDigit(c))
                {
                    _i++;
                    continue;
                }
                if (c is '.' or ',')
                {
                    _i++;
                    continue;
                }
                break;
            }

            if (start == _i)
                throw new FormatException($"Expected number at pos {_i}");

            string token = _s.Substring(start, _i - start);
            token = token.Replace(',', '.');
            if (!double.TryParse(token, System.Globalization.NumberStyles.Float, System.Globalization.CultureInfo.InvariantCulture, out double v))
                throw new FormatException($"Bad number '{token}'");

            return new ExpressionNode { Operation = "const", Constant = v };
        }

        private string? TryReadIdentifier()
        {
            SkipWs();
            int start = _i;
            if (_i >= _s.Length) return null;
            char c0 = _s[_i];
            if (!char.IsLetter(c0)) return null;
            _i++;
            while (_i < _s.Length)
            {
                char c = _s[_i];
                if (char.IsLetterOrDigit(c) || c == '_')
                    _i++;
                else
                    break;
            }
            return _s.Substring(start, _i - start);
        }

        private void SkipWs()
        {
            while (_i < _s.Length && char.IsWhiteSpace(_s[_i])) _i++;
        }

        private bool TryConsume(char c)
        {
            SkipWs();
            if (_i < _s.Length && _s[_i] == c)
            {
                _i++;
                return true;
            }
            return false;
        }

        private void Expect(char c)
        {
            SkipWs();
            if (_i >= _s.Length || _s[_i] != c)
                throw new FormatException($"Expected '{c}' at pos {_i}");
            _i++;
        }
    }

    // Узел дерева выражений
    public class ExpressionNode
    {
        public string Operation { get; set; }
        public List<ExpressionNode> Children { get; set; }
        public double? Constant { get; set; }
        public int? VariableIndex { get; set; } 
        public int? BaseFunctionIndex { get; set; } // индекс пользовательской базовой функции
        public int Depth { get; set; }

        public ExpressionNode()
        {
            Children = new List<ExpressionNode>();
        }

        public ExpressionNode Clone()
        {
            var node = new ExpressionNode
            {
                Operation = this.Operation,
                Constant = this.Constant,
                VariableIndex = this.VariableIndex,
                BaseFunctionIndex = this.BaseFunctionIndex,
                Depth = this.Depth
            };
            
            foreach (var child in Children)
            {
                node.Children.Add(child.Clone());
            }
            
            return node;
        }

        public override string ToString()
        {
            if (Operation == "const")
                return Constant?.ToString("F4") ?? "0";
            if (Operation == "var")
                return $"x{(VariableIndex ?? 0) + 1}";
            if (Operation == "base")
                return $"f{(BaseFunctionIndex ?? 0) + 1}";
            if (Operation == "pow")
                return $"({Children[0]})^{Children[1]}";
            if (Operation == "neg")
                return $"-({Children[0]})";
            
            switch (Operation)
            {
                case "+":
                    if (Children.Count < 2) return "+";
                    return $"({Children[0]} + {Children[1]})";
                case "-":
                    if (Children.Count < 2) return "-";
                    return $"({Children[0]} - {Children[1]})";
                case "*":
                    if (Children.Count < 2) return "*";
                    return $"({Children[0]} * {Children[1]})";
                case "/":
                    if (Children.Count < 2) return "/";
                    return $"({Children[0]} / {Children[1]})";
                case "sin":
                    if (Children.Count < 1) return "sin(?)";
                    return $"sin({Children[0]})";
                case "cos":
                    if (Children.Count < 1) return "cos(?)";
                    return $"cos({Children[0]})";
                case "exp":
                    if (Children.Count < 1) return "exp(?)";
                    return $"exp({Children[0]})";
                case "log":
                    if (Children.Count < 1) return "log(?)";
                    return $"log({Children[0]})";
                default:
                    return Operation;
            }
        }
    }
}
