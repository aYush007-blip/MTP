import re


GREEK = {
    "alpha": r"\alpha",
    "beta": r"\beta",
    "gamma": r"\gamma",
    "delta": r"\delta",
    "epsilon": r"\epsilon",
    "theta": r"\theta",
    "lambda": r"\lambda",
    "mu": r"\mu",
    "sigma": r"\sigma",
    "omega": r"\omega",
}


def normalize(text):
    text = text.lower().strip()
    text = re.sub(r"[.,!?;:]+$", "", text)
    text = re.sub(r"\s+", " ", text)
    return text


def parse_summation(text):
    pattern = (
        r"(\w+)\s+equals\s+(?:the\s+)?"
        r"(?:sum|summation)\s+from\s+(\w+)\s+equals\s+(\w+)\s+"
        r"to\s+(\w+)\s+of\s+(\w+)(?:\s+(?:sub\s+)?(\w+))?"
    )

    match = re.search(pattern, text)

    if not match:
        return None

    lhs, index, lower, upper, variable, subscript = match.groups()

    lhs = GREEK.get(lhs, lhs)

    term = variable
    if subscript:
        term += f"_{{{subscript}}}"

    return (
        f"{lhs}=\\sum_{{{index}={lower}}}^{{{upper}}}{term}"
    )


def parse_engineering_equation(text):
    patterns = [
        (
            r"voltage\s+(?:is\s+)?equal\s+to\s+"
            r"current\s+(?:multiplied\s+by|times)\s+resistance",
            r"V=IR"
        ),
        (
            r"power\s+(?:is\s+)?equal\s+to\s+"
            r"voltage\s+(?:multiplied\s+by|times)\s+current",
            r"P=VI"
        ),
        (
            r"force\s+(?:is\s+)?equal\s+to\s+"
            r"mass\s+(?:multiplied\s+by|times)\s+acceleration",
            r"F=ma"
        ),
        (r"v\s+(?:equals|is equal to)\s+i\s+times\s+r", r"V=IR"),
        (r"p\s+(?:equals|is equal to)\s+v\s+times\s+i", r"P=VI"),
        (r"f\s+(?:equals|is equal to)\s+m\s+times\s+a", r"F=ma"),
    ]

    for pattern, result in patterns:
        if re.fullmatch(pattern, text):
            return result

    return None


def parse(text):
    text = normalize(text)
    result = parse_engineering_equation(text)

    if result:
        return result

    result = parse_summation(text)

    if result:
        return result

    return text


if __name__ == "__main__":
    examples = [
        "sigma equals summation from i equals 1 to n of x i",
        "v equals i times r",
        "voltage is equal to current multiplied by resistance",
    ]

    for text in examples:
        print("Input :", text)
        print("Output:", parse(text))
