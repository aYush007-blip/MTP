"""
Synthetic Mathematical Speech-to-LaTeX Dataset Generator
Generates realistic, diverse paired spoken-math inputs and canonical LaTeX targets.

Key design criteria:
1. Realistic natural phrasing (no artificial 'open parenthesis ... close parenthesis').
2. Dual representation: both spoken words ('x squared plus three x') and Whisper-style digits ('x squared plus 3x').
3. Strict guardrails against isolated number words ('nine' -> '9' is completely barred).
4. Balanced braces and canonical, clean LaTeX targets.
5. Covers arithmetic, algebra, fractions, powers, roots, trigonometry, logarithms,
   derivatives, integrals, summations, limits, and Greek symbols.
"""

import json
import random
import re
from pathlib import Path

SEED = 42
TOTAL_SAMPLES = 10000
rng = random.Random(SEED)

OUT_FILE = Path("/mnt/thesis/datasets/MathSpeech/synthetic.jsonl")
OUT_FILE.parent.mkdir(parents=True, exist_ok=True)

# Common variables and constants
VARIABLES = ["x", "y", "z", "a", "b", "c", "m", "n", "t", "u", "v", "r", "k", "p", "q"]
GREEK_SYMBOLS = [
    ("alpha", r"\alpha"),
    ("beta", r"\beta"),
    ("gamma", r"\gamma"),
    ("delta", r"\delta"),
    ("theta", r"\theta"),
    ("lambda", r"\lambda"),
    ("mu", r"\mu"),
    ("pi", r"\pi"),
    ("sigma", r"\sigma"),
    ("omega", r"\omega"),
    ("phi", r"\phi"),
]

NUMBER_MAP = {
    0: ("zero", "0"),
    1: ("one", "1"),
    2: ("two", "2"),
    3: ("three", "3"),
    4: ("four", "4"),
    5: ("five", "5"),
    6: ("six", "6"),
    7: ("seven", "7"),
    8: ("eight", "8"),
    9: ("nine", "9"),
    10: ("ten", "10"),
    12: ("twelve", "12"),
    15: ("fifteen", "15"),
    20: ("twenty", "20"),
    50: ("fifty", "50"),
    100: ("one hundred", "100"),
}


def pick_num(min_val=0, max_val=10, allow_digits=True):
    val = rng.randint(min_val, max_val)
    word, digit = NUMBER_MAP.get(val, (str(val), str(val)))
    if allow_digits and rng.random() < 0.6:
        return digit, digit
    return word, digit


def pick_var(include_greek=True):
    if include_greek and rng.random() < 0.2:
        return rng.choice(GREEK_SYMBOLS)
    v = rng.choice(VARIABLES)
    return v, v


def pick_coeff(allow_digits=True):
    val = rng.randint(2, 9)
    word, digit = NUMBER_MAP[val]
    var_spk, var_tex = pick_var(include_greek=False)
    style = rng.random()
    if style < 0.4:  # e.g., "3x"
        return f"{digit}{var_spk}", f"{digit}{var_tex}"
    elif style < 0.7:  # e.g., "3 x"
        return f"{digit} {var_spk}", f"{digit}{var_tex}"
    else:  # e.g., "three x"
        return f"{word} {var_spk}", f"{digit}{var_tex}"


# =====================================================================
# Generators for distinct mathematical categories
# =====================================================================

def gen_linear_equation():
    """e.g. 2x + 3 = 7, ax + b = c, y = mx + c"""
    var_spk, var_tex = pick_var(include_greek=False)
    c_spk, c_tex = pick_coeff()
    n1_spk, n1_tex = pick_num(1, 10)
    n2_spk, n2_tex = pick_num(1, 20)
    op_spk, op_tex = rng.choice([("plus", "+"), ("minus", "-")])
    eq_spk = rng.choice(["equals", "is equal to"])

    patterns = [
        (f"{c_spk} {op_spk} {n1_spk} {eq_spk} {n2_spk}",
         f"{c_tex} {op_tex} {n1_tex} = {n2_tex}"),
        (f"{var_spk} {op_spk} {n1_spk} {eq_spk} {n2_spk}",
         f"{var_tex} {op_tex} {n1_tex} = {n2_tex}"),
        (f"y {eq_spk} {c_spk} {op_spk} {n1_spk}",
         f"y = {c_tex} {op_tex} {n1_tex}"),
    ]
    return rng.choice(patterns)


def gen_polynomial():
    """e.g. x squared plus 5x plus 6 equals zero"""
    var_spk, var_tex = pick_var(include_greek=False)
    p2_spk = rng.choice(["squared", "to the power 2", "to the second power"])
    
    coeff = pick_num(2, 9)
    if rng.random() < 0.5:
        c_spk = f"{coeff[0]} {var_spk}"
        c_tex = f"{coeff[1]}{var_tex}"
    else:
        c_spk = f"{var_spk}"
        c_tex = f"{var_tex}"
        
    n_spk, n_tex = pick_num(1, 12)
    eq_spk = rng.choice(["equals", "is equal to"])
    
    rhs_choice = rng.random()
    if rhs_choice < 0.5:
        rhs_spk, rhs_tex = "zero", "0"
    elif rhs_choice < 0.75:
        rhs_spk, rhs_tex = "0", "0"
    else:
        rhs_spk, rhs_tex = pick_num(1, 10)
        
    op1_spk, op1_tex = rng.choice([("plus", "+"), ("minus", "-")])
    op2_spk, op2_tex = rng.choice([("plus", "+"), ("minus", "-")])

    if rng.random() < 0.7:
        spk = f"{var_spk} {p2_spk} {op1_spk} {c_spk} {op2_spk} {n_spk} {eq_spk} {rhs_spk}"
        tex = f"{var_tex}^{{2}} {op1_tex} {c_tex} {op2_tex} {n_tex} = {rhs_tex}"
    else:
        spk = f"{var_spk} {p2_spk} {op1_spk} {n_spk}"
        tex = f"{var_tex}^{{2}} {op1_tex} {n_tex}"
    return spk, tex


def gen_power_exponent():
    """e.g. x cubed, x to the power n, e to the minus t, 10 to the power 5"""
    var_spk, var_tex = pick_var(include_greek=True)
    pow_val = rng.randint(2, 6)
    w_spk, d_spk = NUMBER_MAP[pow_val]

    pow_types = [
        ("squared", "2"),
        ("cubed", "3"),
        (f"to the power {d_spk}", f"{pow_val}"),
        (f"to the power of {w_spk}", f"{pow_val}"),
        (f"to the {d_spk} th power", f"{pow_val}"),
        ("to the power n", "n"),
        ("to the power minus one", "-1"),
        ("to the minus 2", "-2"),
    ]
    spk_pow, tex_pow = rng.choice(pow_types)

    if rng.random() < 0.2:
        # e^x or e^{-t}
        t_spk, t_tex = rng.choice([("x", "x"), ("t", "t"), ("minus t", "-t"), ("minus x", "-x"), ("2t", "2t")])
        e_spk = rng.choice([f"e to the power {t_spk}", f"e to the {t_spk}", f"exponential of {t_spk}"])
        return e_spk, f"e^{{{t_tex}}}"

    return f"{var_spk} {spk_pow}", f"{var_tex}^{{{tex_pow}}}"


def gen_fraction():
    """e.g. a over b, 1 over x, x plus 1 over x minus 1, one half of x"""
    v1_spk, v1_tex = pick_var(include_greek=True)
    v2_spk, v2_tex = pick_var(include_greek=True)
    while v2_tex == v1_tex:
        v2_spk, v2_tex = pick_var(include_greek=True)

    n1_spk, n1_tex = pick_num(1, 9)
    n2_spk, n2_tex = pick_num(2, 9)

    frac_type = rng.random()
    if frac_type < 0.25:
        # Simple variable fraction
        over = rng.choice(["over", "divided by"])
        return f"{v1_spk} {over} {v2_spk}", rf"\frac{{{v1_tex}}}{{{v2_tex}}}"
    elif frac_type < 0.5:
        # Number over variable or variable over number
        if rng.random() < 0.5:
            return f"{n1_spk} over {v2_spk}", rf"\frac{{{n1_tex}}}{{{v2_tex}}}"
        else:
            return f"{v1_spk} over {n2_spk}", rf"\frac{{{v1_tex}}}{{{n2_tex}}}"
    elif frac_type < 0.75:
        # Polynomial fraction: (x + 1) / (x - 1)
        op1_spk, op1_tex = rng.choice([("plus", "+"), ("minus", "-")])
        op2_spk, op2_tex = rng.choice([("plus", "+"), ("minus", "-")])
        over = rng.choice(["over", "all over", "divided by"])
        spk = f"{v1_spk} {op1_spk} {n1_spk} {over} {v1_spk} {op2_spk} {n2_spk}"
        tex = rf"\frac{{{v1_tex} {op1_tex} {n1_tex}}}{{{v1_tex} {op2_tex} {n2_tex}}}"
        return spk, tex
    else:
        # One half / one third style
        coef_spk, coef_tex = rng.choice([
            ("one half", r"\frac{1}{2}"),
            ("1 over 2", r"\frac{1}{2}"),
            ("one third", r"\frac{1}{3}"),
            ("one fourth", r"\frac{1}{4}"),
            ("three fourths", r"\frac{3}{4}"),
        ])
        return f"{coef_spk} {v1_spk}", f"{coef_tex}{v1_tex}"


def gen_root():
    """e.g. square root of x, square root of a squared plus b squared"""
    v1_spk, v1_tex = pick_var(include_greek=True)
    v2_spk, v2_tex = pick_var(include_greek=False)
    n_spk, n_tex = pick_num(2, 16)

    root_type = rng.random()
    if root_type < 0.4:
        return f"square root of {v1_spk}", rf"\sqrt{{{v1_tex}}}"
    elif root_type < 0.65:
        return f"square root of {n_spk}", rf"\sqrt{{{n_tex}}}"
    elif root_type < 0.85:
        op_spk, op_tex = rng.choice([("plus", "+"), ("minus", "-")])
        spk = f"square root of {v1_spk} squared {op_spk} {v2_spk} squared"
        tex = rf"\sqrt{{{v1_tex}^{{2}} {op_tex} {v2_tex}^{{2}}}}"
        return spk, tex
    else:
        return f"cube root of {v1_spk}", rf"\sqrt[3]{{{v1_tex}}}"


def gen_trigonometry():
    """e.g. sin x, cosine of theta, sin squared theta plus cos squared theta equals 1"""
    func_spk, func_tex = rng.choice([
        ("sine of", r"\sin"),
        ("sin", r"\sin"),
        ("cosine of", r"\cos"),
        ("cos", r"\cos"),
        ("tangent of", r"\tan"),
        ("tan", r"\tan"),
        ("secant of", r"\sec"),
        ("cotangent of", r"\cot"),
    ])
    var_spk, var_tex = pick_var(include_greek=True)

    trig_type = rng.random()
    if trig_type < 0.5:
        return f"{func_spk} {var_spk}", rf"{func_tex}\left({var_tex}\right)"
    elif trig_type < 0.75:
        # e.g., cos 2 theta or sin omega t
        c_spk, c_tex = rng.choice([("two", "2"), ("2", "2"), ("omega", r"\omega")])
        if c_spk == "omega":
            return f"sine of omega t", rf"\sin\left(\omega t\right)"
        return f"{func_spk} {c_spk} {var_spk}", rf"{func_tex}\left({c_tex}{var_tex}\right)"
    else:
        # Identity: sin^2(x) + cos^2(x) = 1
        return (f"sine squared {var_spk} plus cosine squared {var_spk} equals 1",
                rf"\sin^{{2}}\left({var_tex}\right) + \cos^{{2}}\left({var_tex}\right) = 1")


def gen_log_exp():
    """e.g. natural log of x, ln x, log base 10 of x"""
    var_spk, var_tex = pick_var(include_greek=False)
    log_type = rng.random()
    if log_type < 0.4:
        phr = rng.choice(["natural log of", "ln of", "ln"])
        return f"{phr} {var_spk}", rf"\ln\left({var_tex}\right)"
    elif log_type < 0.7:
        phr = rng.choice(["log of", "log"])
        return f"{phr} {var_spk}", rf"\log\left({var_tex}\right)"
    else:
        return f"log base 10 of {var_spk}", rf"\log_{{10}}\left({var_tex}\right)"


def gen_calculus_derivative():
    """e.g. dy by dx, derivative of y with respect to x, partial u over partial t"""
    deriv_type = rng.random()
    if deriv_type < 0.4:
        v1, v2 = rng.choice([("y", "x"), ("u", "t"), ("v", "t"), ("s", "t")])
        phr = rng.choice([
            f"d {v1} by d {v2}",
            f"d {v1} over d {v2}",
            f"derivative of {v1} with respect to {v2}"
        ])
        return phr, rf"\frac{{d{v1}}}{{d{v2}}}"
    elif deriv_type < 0.7:
        # Second derivative
        v1, v2 = rng.choice([("y", "x"), ("u", "t")])
        phr = rng.choice([
            f"second derivative of {v1} with respect to {v2}",
            f"d 2 {v1} by d {v2} squared",
            f"d squared {v1} over d {v2} squared",
        ])
        return phr, rf"\frac{{d^{{2}}{v1}}}{{d{v2}^{{2}}}}"
    else:
        # Partial derivative
        u, x = rng.choice([("u", "x"), ("f", "x"), ("u", "t"), ("psi", "t")])
        u_tex = r"\psi" if u == "psi" else u
        phr = f"partial {u} over partial {x}"
        return phr, rf"\frac{{\partial {u_tex}}}{{\partial {x}}}"


def gen_calculus_integral():
    """e.g. integral of x dx, integral from 0 to 1 of x squared dx"""
    var_spk, var_tex = pick_var(include_greek=False)
    int_type = rng.random()
    if int_type < 0.5:
        expr_spk, expr_tex = rng.choice([
            (var_spk, var_tex),
            (f"{var_spk} squared", f"{var_tex}^{{2}}"),
            (f"sine of {var_spk}", rf"\sin\left({var_tex}\right)"),
            (f"e to the {var_spk}", f"e^{{{var_tex}}}"),
        ])
        return f"integral of {expr_spk} d {var_spk}", rf"\int {expr_tex} \, d{var_tex}"
    else:
        # Definite integral
        l_spk, l_tex = rng.choice([("zero", "0"), ("0", "0"), ("minus infinity", r"-\infty")])
        u_spk, u_tex = rng.choice([("one", "1"), ("1", "1"), ("infinity", r"\infty"), ("t", "t")])
        return (f"integral from {l_spk} to {u_spk} of {var_spk} d {var_spk}",
                rf"\int_{{{l_tex}}}^{{{u_tex}}} {var_tex} \, d{var_tex}")


def gen_summation():
    """e.g. sum from i equals 1 to n of x sub i"""
    idx_spk, idx_tex = rng.choice([("i", "i"), ("k", "k"), ("n", "n")])
    lower_spk, lower_tex = rng.choice([("1", "1"), ("one", "1"), ("zero", "0"), ("0", "0")])
    upper_spk, upper_tex = rng.choice([("n", "n"), ("N", "N"), ("infinity", r"\infty"), ("10", "10")])
    term_spk, term_tex = rng.choice([
        (f"x sub {idx_spk}", f"x_{{{idx_tex}}}"),
        (f"a sub {idx_spk}", f"a_{{{idx_tex}}}"),
        (f"x sub {idx_spk} squared", f"x_{{{idx_tex}}}^{{2}}"),
        (f"1 over {idx_spk}", rf"\frac{{1}}{{{idx_tex}}}"),
    ])
    sum_phr = rng.choice(["sum", "summation"])
    spk = f"{sum_phr} from {idx_spk} equals {lower_spk} to {upper_spk} of {term_spk}"
    tex = rf"\sum_{{{idx_tex}={lower_tex}}}^{{{upper_tex}}} {term_tex}"
    return spk, tex


def gen_limit():
    """e.g. limit as x approaches 0 of sin x over x equals 1"""
    var_spk, var_tex = ("x", "x")
    appr_spk, appr_tex = rng.choice([
        ("0", "0"),
        ("zero", "0"),
        ("infinity", r"\infty"),
    ])
    if appr_tex == "0":
        return ("limit as x approaches 0 of sine of x over x equals 1",
                r"\lim_{x \to 0} \frac{\sin\left(x\right)}{x} = 1")
    else:
        return ("limit as n approaches infinity of 1 over n equals 0",
                r"\lim_{n \to \infty} \frac{1}{n} = 0")


def gen_subscript_relation():
    """e.g. x sub 1 plus x sub 2 equals 10, v sub 0 plus a t"""
    v_spk, v_tex = pick_var(include_greek=False)
    i1_spk, i1_tex = ("1", "1") if rng.random() < 0.5 else ("zero", "0")
    i2_spk, i2_tex = ("2", "2") if rng.random() < 0.5 else ("one", "1")
    op_spk, op_tex = rng.choice([("plus", "+"), ("minus", "-")])
    n_spk, n_tex = pick_num(1, 10)

    return (f"{v_spk} sub {i1_spk} {op_spk} {v_spk} sub {i2_spk} equals {n_spk}",
            f"{v_tex}_{{{i1_tex}}} {op_tex} {v_tex}_{{{i2_tex}}} = {n_tex}")


def gen_physics_math_laws():
    """Standard scientific equations with diverse spoken variants."""
    laws = [
        # Ohm's Law
        [("v equals i times r", "V = IR"),
         ("v equals i r", "V = IR"),
         ("voltage equals current times resistance", "V = IR"),
         ("i equals v over r", r"I = \frac{V}{R}")],
        # Power
        [("p equals v times i", "P = VI"),
         ("p equals i squared r", "P = I^{2}R"),
         ("p equals v squared over r", r"P = \frac{V^{2}}{R}")],
        # Newton / Mechanics
        [("f equals m times a", "F = ma"),
         ("f equals m a", "F = ma"),
         ("force equals mass times acceleration", "F = ma"),
         ("kinetic energy equals one half m v squared", r"E_k = \frac{1}{2}mv^{2}"),
         ("e equals one half m v squared", r"E = \frac{1}{2}mv^{2}"),
         ("p equals m times v", "p = mv")],
        # Einstein
        [("e equals m c squared", "E = mc^{2}"),
         ("energy equals m c squared", "E = mc^{2}")],
        # Kinematics
        [("v equals u plus a t", "v = u + at"),
         ("s equals u t plus one half a t squared", r"s = ut + \frac{1}{2}at^{2}"),
         ("v squared equals u squared plus 2 a s", "v^{2} = u^{2} + 2as")],
        # Wave / Circle / Geometry
        [("v equals f times lambda", r"v = f\lambda"),
         ("c equals 2 pi r", r"C = 2\pi r"),
         ("a equals pi r squared", r"A = \pi r^{2}"),
         ("area equals pi r squared", r"A = \pi r^{2}")],
        # Capacitance & Inductance
        [("q equals c times v", "Q = CV"),
         ("e equals one half c v squared", r"E = \frac{1}{2}CV^{2}"),
         ("e equals one half l i squared", r"E = \frac{1}{2}LI^{2}")],
    ]
    family = rng.choice(laws)
    return rng.choice(family)


# Master dispatch table with probabilities
GENERATORS = [
    (gen_linear_equation, 0.15),
    (gen_polynomial, 0.15),
    (gen_power_exponent, 0.12),
    (gen_fraction, 0.14),
    (gen_root, 0.08),
    (gen_trigonometry, 0.10),
    (gen_log_exp, 0.06),
    (gen_calculus_derivative, 0.06),
    (gen_calculus_integral, 0.05),
    (gen_summation, 0.04),
    (gen_limit, 0.02),
    (gen_subscript_relation, 0.05),
    (gen_physics_math_laws, 0.08),
]


def generate_single_example():
    """Picks a generator according to weights and applies Whisper post-processing."""
    r = rng.random()
    cum = 0.0
    gen_fn = GENERATORS[0][0]
    for fn, weight in GENERATORS:
        cum += weight
        if r <= cum:
            gen_fn = fn
            break

    spoken, latex = gen_fn()

    # Clean whitespace
    spoken = " ".join(spoken.strip().split())
    latex = " ".join(latex.strip().split())

    # Ensure LaTeX tokens are properly spaced where required
    latex = re.sub(r"\s+", " ", latex)

    return {
        "input": spoken,
        "target": latex,
        "source": "synthetic_v2",
    }


def validate_example(example):
    """Rigorous sanity checks to prevent training corruption."""
    inp = example["input"].strip()
    tgt = example["target"].strip()

    if not inp or not tgt:
        return False, "Empty input or target"

    # Guardrail: Never allow a single isolated word number (e.g. "nine" -> "9")
    if inp.lower() in [w for w, _ in NUMBER_MAP.values()]:
        return False, f"Isolated number forbidden: {inp}"

    # Guardrail: Balanced braces in target
    if tgt.count("{") != tgt.count("}"):
        return False, f"Unbalanced braces in target: {tgt}"

    # Guardrail: Balanced parentheses in target
    if tgt.count("(") != tgt.count(")"):
        return False, f"Unbalanced parens in target: {tgt}"

    return True, "OK"


def build_dataset(total_records=TOTAL_SAMPLES):
    records = []
    seen = set()
    attempts = 0
    max_attempts = total_records * 5

    print(f"Generating {total_records} synthetic math-to-LaTeX pairs...")

    while len(records) < total_records and attempts < max_attempts:
        attempts += 1
        ex = generate_single_example()
        key = (ex["input"].lower(), ex["target"])

        if key in seen:
            continue

        valid, reason = validate_example(ex)
        if not valid:
            continue

        seen.add(key)
        records.append(ex)

    with OUT_FILE.open("w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    print(f"Successfully generated {len(records)} examples across {attempts} attempts.")
    print(f"Saved to: {OUT_FILE}")
    print("\nSample pairs from generated dataset:")
    for sample in records[:15]:
        print(f"  Input : {sample['input']}")
        print(f"  Target: {sample['target']}\n")


if __name__ == "__main__":
    build_dataset()
