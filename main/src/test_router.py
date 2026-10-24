"""
Comprehensive Benchmark & Verification Suite for MathRouter.
Validates:
1. 0% false positives on Pure English sentences.
2. 0% false positives on Descriptive Physics/Engineering English sentences.
3. 0% false positives on Conversational English with counts and numbers.
4. Faithful classification and translation of Pure Math formulas.
5. Accurate span extraction and LaTeX replacement on Mixed utterances.
"""

from router import MathRouter, RouteType, T5MathTranslator


def run_benchmark():
    # Test cases categorized by expected behavior
    test_suite = [
        # Track 1: Pure Conversational English
        {
            "category": "pure_english",
            "expected_route": RouteType.PURE_ENGLISH,
            "inputs": [
                "I am testing the live speech transcription system.",
                "The microphone captures my voice and sends it to the computer.",
                "The system converts spoken sentences into written text.",
                "This project runs speech recognition offline on a Jetson device.",
                "The voltage across a resistor depends on the current and the resistance.",
                "The experiment measures the output signal at different frequencies.",
                "Please record this sentence clearly and transcribe every word.",
                "A stable power supply is important for reliable measurements.",
                "The real-time audio pipeline detects speech and saves each utterance.",
                "Today I am checking the accuracy and response time of the system.",
            ],
        },
        # Track 2: English with Numbers / Counts / Dates (Must remain English!)
        {
            "category": "english_with_numbers",
            "expected_route": RouteType.PURE_ENGLISH,
            "inputs": [
                "There are nine students in room forty five.",
                "The meeting is at two o'clock on October 9th.",
                "Chapter three has twelve exercises.",
                "We tested twenty samples across two devices.",
            ],
        },
        # Track 3: Descriptive Physics / Engineering English (Must remain English!)
        {
            "category": "descriptive_physics",
            "expected_route": RouteType.PURE_ENGLISH,
            "inputs": [
                "Voltage is the product of resistance and current.",
                "Power is the product of voltage and current.",
                "Force is the product of mass and acceleration.",
                "The current through the resistor increases when the applied voltage increases.",
                "The total resistance in a series circuit is the sum of the individual resistances.",
                "The output voltage depends on the input signal and the gain of the amplifier.",
                "The frequency is measured in hertz and the phase is measured in degrees.",
                "The energy stored in a capacitor depends on its capacitance and voltage.",
            ],
        },
        # Track 4: Pure Symbolic Math Formulas (Must route to PURE_MATH and generate LaTeX)
        {
            "category": "pure_math",
            "expected_route": RouteType.PURE_MATH,
            "inputs": [
                "V equals I times R",
                "P equals V times I",
                "F equals m times a",
                "Sigma equals the summation from i equals 1 to n of x sub i",
                "2b over n",
                "x squared plus y squared equals r squared",
                "two x plus three y equals seven",
            ],
        },
        # Track 5: Mixed Utterances (Must route to MIXED and convert only math span)
        {
            "category": "mixed",
            "expected_route": RouteType.MIXED,
            "inputs": [
                "We know that x plus y equals ten so we can solve for y.",
                "Suppose f of x equals three x squared plus two.",
                "Let y equal sine of theta.",
                "Therefore a squared plus b squared equals c squared.",
                "Notice that two x minus four is greater than zero.",
            ],
        },
    ]

    print("=" * 70)
    print("BENCHMARKING SPOKEN-MATH-TO-LATEX ROUTER & SPAN DETECTOR")
    print("=" * 70)

    # First test classification without loading model (zero-shot linguistic routing)
    fast_router = MathRouter(translator=None)

    total_tests = 0
    passed_tests = 0

    for track in test_suite:
        category = track["category"]
        expected = track["expected_route"]
        inputs = track["inputs"]

        print(f"\n--- Testing Category: {category} (Expected: {expected.value}) ---")
        category_passed = 0

        for text in inputs:
            total_tests += 1
            route, spans = fast_router.classify(text)
            ok = route == expected

            if ok:
                passed_tests += 1
                category_passed += 1
                span_info = f" [Span: '{spans[0][2]}']" if spans else ""
                print(f"  [PASS] {text}{span_info}")
            else:
                span_info = f" [Detected Spans: {spans}]" if spans else " [No spans]"
                print(f"  [FAIL] {text}")
                print(f"         Got: {route.value}, Expected: {expected.value}{span_info}")

        pct = 100 * category_passed / len(inputs)
        print(f"  --> Category Accuracy: {category_passed}/{len(inputs)} ({pct:.1f}%)")

    print("\n" + "=" * 70)
    overall_pct = 100 * passed_tests / total_tests
    print(f"Routing Classification Accuracy: {passed_tests}/{total_tests} ({overall_pct:.1f}%)")
    print("=" * 70)

    # Now run end-to-end with T5 translation
    print("\nTesting End-to-End Execution with T5-v2 Model...")
    translator = T5MathTranslator.get_instance()
    full_router = MathRouter(translator=translator)

    e2e_samples = [
        "There are nine students in room forty five.",
        "Voltage is the product of resistance and current.",
        "V equals I times R",
        "2b over n",
        "We know that x plus y equals ten so we can solve for y.",
        "Suppose f of x equals three x squared plus two.",
    ]

    print("\n" + "-" * 70)
    for sample in e2e_samples:
        result = full_router.process(sample)
        print(f"Input:     {sample}")
        print(f"Route:     {result.route_type.value}")
        print(f"Processed: {result.processed_text}")
        if result.spans:
            print(f"Spans:     {[s.spoken_text + ' -> ' + s.latex_text for s in result.spans]}")
        print("-" * 70)


if __name__ == "__main__":
    run_benchmark()
