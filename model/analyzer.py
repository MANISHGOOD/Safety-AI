import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from life_saving_rules import detect_life_saving_rule
from hazard_extractor import extract_hazard_barrier_activity


# ============================================================
# SIH26165 - UNIFIED SAFETY REPORT ANALYZER
# ============================================================

DATASET_PATH = "C:/SIH26165/dataset/safety_reports.csv"


print("=" * 70)
print("SIH26165 - UNIFIED SAFETY REPORT ANALYZER")
print("=" * 70)


# ============================================================
# 1. LOAD DATASET
# ============================================================

print("\nLoading dataset...")

data = pd.read_csv(DATASET_PATH)

print("Dataset loaded successfully.")
print("Total reports:", len(data))


# ============================================================
# 2. CHECK REQUIRED COLUMNS
# ============================================================

required_columns = [
    "report_text",
    "sif_potential"
]

for column in required_columns:

    if column not in data.columns:

        raise ValueError(
            f"Required column '{column}' was not found in the dataset."
        )


# ============================================================
# 3. PREPARE TRAINING DATA
# ============================================================

X = data["report_text"].astype(str)

y = data["sif_potential"].astype(str)


# ============================================================
# 4. CREATE NLP MODEL
# ============================================================

print("\nTraining SIF classification model...")

model = Pipeline([

    (
        "tfidf",
        TfidfVectorizer(
            lowercase=True,
            stop_words="english",
            ngram_range=(1, 2),
            min_df=1
        )
    ),

    (
        "classifier",
        LogisticRegression(
            max_iter=2000,
            random_state=42
        )
    )
])


# ============================================================
# 5. TRAIN MODEL
# ============================================================

model.fit(X, y)

print("SIF model trained successfully.")


# ============================================================
# 6. ANALYZE ONE SAFETY REPORT
# ============================================================

def analyze_report(report):

    report = str(report).strip()

    # --------------------------------------------------------
    # Empty report
    # --------------------------------------------------------

    if not report:

        return {
            "report": "",
            "sif_potential": "UNKNOWN",
            "sif_confidence": 0.0,

            "life_saving_rule":
                "No clear Life-Saving Rule detected",

            "rule_keyword_matches": 0,

            "activity": "Not detected",
            "hazard": "Not detected",
            "barrier_failure": "Not detected",

            "hazard_keyword_matches": 0,
            "barrier_keyword_matches": 0,
            "activity_keyword_matches": 0
        }


    # --------------------------------------------------------
    # SIF prediction
    # --------------------------------------------------------

    prediction = model.predict([report])[0]

    probabilities = model.predict_proba([report])[0]

    confidence = max(probabilities) * 100


    # --------------------------------------------------------
    # Life-Saving Rule detection
    # --------------------------------------------------------

    rule, rule_score = detect_life_saving_rule(report)


    # --------------------------------------------------------
    # Hazard / Barrier / Activity extraction
    # --------------------------------------------------------

    details = extract_hazard_barrier_activity(report)


    # --------------------------------------------------------
    # Return combined result
    # --------------------------------------------------------

    result = {

        "report": report,

        "sif_potential": prediction,

        "sif_confidence": round(confidence, 2),

        "life_saving_rule": rule,

        "rule_keyword_matches": rule_score,

        "activity": details["activity"],

        "hazard": details["hazard"],

        "barrier_failure": details["barrier_failure"],

        "hazard_keyword_matches":
            details["hazard_keyword_matches"],

        "barrier_keyword_matches":
            details["barrier_keyword_matches"],

        "activity_keyword_matches":
            details["activity_keyword_matches"]
    }


    return result


# ============================================================
# 7. DISPLAY ANALYSIS RESULT
# ============================================================

def print_analysis(result):

    print("\n" + "=" * 70)
    print("SAFETY REPORT ANALYSIS")
    print("=" * 70)

    print("\nReport:")
    print(result["report"])

    print("\nSIF Potential:")
    print(result["sif_potential"])

    print("\nSIF Model Confidence:")
    print(str(result["sif_confidence"]) + "%")

    print("\nLife-Saving Rule:")
    print(result["life_saving_rule"])

    print("\nRule Keyword Matches:")
    print(result["rule_keyword_matches"])

    print("\nActivity:")
    print(result["activity"])

    print("\nHazard:")
    print(result["hazard"])

    print("\nBarrier Failure:")
    print(result["barrier_failure"])

    print("\nHazard Keyword Matches:")
    print(result["hazard_keyword_matches"])

    print("\nBarrier Keyword Matches:")
    print(result["barrier_keyword_matches"])

    print("\nActivity Keyword Matches:")
    print(result["activity_keyword_matches"])

    print("\n" + "=" * 70)


# ============================================================
# 8. TEST REPORTS
# ============================================================

if __name__ == "__main__":

    test_reports = [

        "A technician entered a vessel without checking for toxic gases.",

        "During maintenance the pump unexpectedly started because "
        "proper isolation was not verified.",

        "A worker was standing below a suspended load during crane lifting.",

        "Hot work was performed near flammable material without a gas test.",

        "A worker climbed a platform without using fall protection.",

        "Workers excavated near an underground pipeline without "
        "proper authorization.",

        "A driver was reversing a truck without a spotter.",

        "An electrician opened an energized electrical panel "
        "without isolating the power."
    ]


    print("\n")
    print("=" * 70)
    print("TESTING UNIFIED ANALYZER")
    print("=" * 70)


    for number, report in enumerate(test_reports, start=1):

        print("\n")
        print("#" * 70)
        print(f"TEST REPORT #{number}")
        print("#" * 70)

        result = analyze_report(report)

        print_analysis(result)


    # ========================================================
    # 9. INTERACTIVE MODE
    # ========================================================

    print("\n")
    print("=" * 70)
    print("INTERACTIVE SAFETY REPORT ANALYZER")
    print("=" * 70)

    print("\nEnter a safety report.")

    print("The system will detect:")
    print("  1. SIF potential")
    print("  2. SIF confidence")
    print("  3. Life-Saving Rule")
    print("  4. Activity")
    print("  5. Hazard")
    print("  6. Barrier failure")

    print("\nType 'exit' to stop.")


    while True:

        report = input("\nSafety report: ")


        if report.lower().strip() == "exit":

            break


        if not report.strip():

            print("Please enter a safety report.")

            continue


        result = analyze_report(report)

        print_analysis(result)


    print("\n")
    print("=" * 70)
    print("STEP 5 - UNIFIED ANALYZER COMPLETED")
    print("=" * 70)