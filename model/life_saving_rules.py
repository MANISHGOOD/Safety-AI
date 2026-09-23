import re


# ============================================================
# SIH26165 - LIFE-SAVING RULE DETECTION
# ============================================================

RULES = {

    "Confined Space": [
        "confined space",
        "vessel",
        "tank",
        "manhole",
        "inside vessel",
        "inside tank",
        "gas testing",
        "gas test",
        "toxic gas",
        "oxygen deficiency",
        "standby person",
        "entry permit"
    ],

    "Energy Isolation": [
        "isolation",
        "isolated",
        "lockout",
        "lock out",
        "tagout",
        "lockout tagout",
        "loto",
        "unexpected startup",
        "unexpected start",
        "energized",
        "energy source",
        "pressure release",
        "machine started",
        "equipment started"
    ],

    "Line of Fire": [
        "line of fire",
        "suspended load",
        "struck by",
        "caught between",
        "pinch point",
        "moving equipment",
        "moving vehicle",
        "falling object",
        "crush",
        "trapped",
        "swing radius"
    ],

    "Hot Work": [
        "hot work",
        "welding",
        "welding work",
        "cutting",
        "grinding",
        "flame",
        "spark",
        "ignition",
        "flammable material",
        "fire watch",
        "gas test"
    ],

    "Working at Height": [
        "working at height",
        "height",
        "scaffold",
        "scaffolding",
        "ladder",
        "fall protection",
        "safety harness",
        "harness",
        "unguarded edge",
        "roof",
        "elevated platform"
    ],

    "Ground Disturbance": [
        "excavation",
        "excavating",
        "trenching",
        "trench",
        "digging",
        "ground disturbance",
        "underground cable",
        "underground pipeline",
        "buried pipeline",
        "buried cable"
    ],

    "Lifting Operations": [
        "lifting",
        "lifting operation",
        "crane",
        "crane operation",
        "rigging",
        "rigging operation",
        "sling",
        "lifting equipment",
        "suspended load",
        "load",
        "hoisting"
    ],

    "Driving and Vehicle Safety": [
        "vehicle",
        "driving",
        "driver",
        "seat belt",
        "seatbelt",
        "speeding",
        "reversing",
        "reverse",
        "mobile equipment",
        "truck",
        "forklift"
    ],

    "Electrical Safety": [
        "electrical",
        "electrical panel",
        "electric shock",
        "electric shock hazard",
        "power supply",
        "live wire",
        "live electrical",
        "energized panel",
        "electrical maintenance",
        "electricity"
    ]
}


# ============================================================
# 1. NORMALIZE TEXT
# ============================================================

def clean_text(text):

    text = str(text).lower()

    text = re.sub(r"\s+", " ", text)

    return text.strip()


# ============================================================
# 2. DETECT LIFE-SAVING RULE
# ============================================================

def detect_life_saving_rule(report):

    text = clean_text(report)

    scores = {}

    for rule, keywords in RULES.items():

        score = 0

        for keyword in keywords:

            if keyword in text:
                score += 1

        scores[rule] = score

    best_rule = max(scores, key=scores.get)

    best_score = scores[best_rule]

    if best_score == 0:
        return "No clear Life-Saving Rule detected", 0

    return best_rule, best_score


# ============================================================
# 3. GET ALL RULE SCORES
# ============================================================

def get_rule_scores(report):

    text = clean_text(report)

    scores = {}

    for rule, keywords in RULES.items():

        score = 0

        for keyword in keywords:

            if keyword in text:
                score += 1

        scores[rule] = score

    return scores


# ============================================================
# 4. TESTING SECTION
# ============================================================
# This section runs ONLY when this file is executed directly.
# It will NOT run when analyzer.py imports this file.

if __name__ == "__main__":

    print("=" * 70)
    print("SIH26165 - LIFE-SAVING RULE DETECTOR")
    print("=" * 70)


    # ========================================================
    # TEST REPORTS
    # ========================================================

    test_reports = [

        "A technician entered a vessel without checking for toxic gases.",

        "During maintenance the pump unexpectedly started because proper isolation was not verified.",

        "A worker was standing below a suspended load during crane lifting.",

        "Hot work was performed near flammable material without a gas test.",

        "A worker climbed a platform without using fall protection.",

        "Workers excavated near an underground pipeline without proper authorization.",

        "A driver was reversing a truck without a spotter.",

        "An electrician opened an energized electrical panel without isolating the power."
    ]


    # ========================================================
    # RUN TESTS
    # ========================================================

    print("\nTEST RESULTS")
    print("=" * 70)

    for number, report in enumerate(test_reports, start=1):

        rule, score = detect_life_saving_rule(report)

        print("\n" + "-" * 70)

        print(f"Report #{number}")
        print("Report:", report)
        print("Life-Saving Rule:", rule)
        print("Keyword matches:", score)


    # ========================================================
    # INTERACTIVE MODE
    # ========================================================

    print("\n" + "=" * 70)
    print("INTERACTIVE LIFE-SAVING RULE DETECTION")
    print("=" * 70)

    print("\nEnter a safety report.")
    print("Type 'exit' to stop.")

    while True:

        report = input("\nSafety report: ")

        if report.lower().strip() == "exit":
            break

        if not report.strip():

            print("Please enter a report.")

            continue

        rule, score = detect_life_saving_rule(report)

        print("\nAI Rule Detection")
        print("-" * 30)

        print("Life-Saving Rule:", rule)
        print("Keyword matches:", score)


    print("\n" + "=" * 70)
    print("LIFE-SAVING RULE DETECTOR TEST COMPLETED")
    print("=" * 70)