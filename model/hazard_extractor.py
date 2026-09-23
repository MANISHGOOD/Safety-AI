import re


# ============================================================
# SIH26165 - HAZARD / BARRIER / ACTIVITY EXTRACTOR
# ============================================================


# ============================================================
# 1. HAZARD KEYWORDS
# ============================================================

HAZARDS = {

    "Toxic Atmosphere": [
        "toxic gas",
        "toxic gases",
        "gas leak",
        "gas testing",
        "gas test",
        "oxygen deficiency",
        "oxygen level",
        "harmful gas"
    ],

    "Unexpected Equipment Startup": [
        "unexpected startup",
        "unexpected start",
        "unexpectedly started",
        "machine started",
        "equipment started"
    ],

    "Suspended Load": [
        "suspended load",
        "crane lifting",
        "lifting load",
        "falling load",
        "overhead load"
    ],

    "Fire / Explosion": [
        "hot work",
        "flammable material",
        "flammable",
        "fire",
        "spark",
        "ignition",
        "welding",
        "gas leak"
    ],

    "Fall from Height": [
        "fall protection",
        "working at height",
        "height",
        "scaffold",
        "scaffolding",
        "ladder",
        "unguarded edge",
        "elevated platform"
    ],

    "Underground Utility Damage": [
        "underground pipeline",
        "underground cable",
        "buried pipeline",
        "buried cable",
        "excavation",
        "excavating",
        "trenching",
        "trench"
    ],

    "Vehicle Collision": [
        "vehicle",
        "truck",
        "driver",
        "driving",
        "reversing",
        "reverse",
        "forklift",
        "moving vehicle"
    ],

    "Electrical Shock": [
        "electrical panel",
        "electric shock",
        "live wire",
        "live electrical",
        "energized panel",
        "electrical maintenance",
        "electricity"
    ],

    "Caught Between / Crushing": [
        "pinch point",
        "caught between",
        "crush",
        "trapped",
        "struck by",
        "line of fire"
    ]
}


# ============================================================
# 2. BARRIER FAILURE KEYWORDS
# ============================================================

BARRIERS = {

    "Gas Testing Not Performed": [
        "without gas testing",
        "without gas test",
        "without checking for toxic gases",
        "gas testing not performed",
        "gas test not performed",
        "no gas test",
        "without checking gas"
    ],

    "Energy Isolation Not Verified": [
        "isolation was not verified",
        "isolation not verified",
        "without proper isolation",
        "without isolation",
        "not isolated",
        "isolation failed",
        "lockout not",
        "lockout was not",
        "loto not"
    ],

    "Fall Protection Not Used": [
        "without fall protection",
        "no fall protection",
        "without safety harness",
        "without harness",
        "harness not used",
        "fall protection not used"
    ],

    "Load Control / Exclusion Zone Failure": [
        "standing below a suspended load",
        "below a suspended load",
        "no exclusion zone",
        "without exclusion zone",
        "suspended load",
        "load control"
    ],

    "Hot Work Controls Missing": [
        "without a gas test",
        "without gas test",
        "no fire watch",
        "fire watch not provided",
        "flammable material",
        "without hot work permit"
    ],

    "Excavation Authorization / Utility Check Missing": [
        "without proper authorization",
        "without authorization",
        "no utility check",
        "underground pipeline",
        "underground cable"
    ],

    "Vehicle Spotter Missing": [
        "without a spotter",
        "no spotter",
        "spotter not present",
        "without spotter"
    ],

    "Electrical Isolation Not Performed": [
        "without isolating the power",
        "without isolation",
        "power not isolated",
        "electrical isolation not performed",
        "energized electrical panel"
    ],

    "Standby Person Missing": [
        "no standby person",
        "without standby person",
        "standby person was not present"
    ]
}


# ============================================================
# 3. ACTIVITY KEYWORDS
# ============================================================

ACTIVITIES = {

    "Vessel Entry": [
        "entered a vessel",
        "entered vessel",
        "inside vessel",
        "vessel entry",
        "tank entry",
        "entered a tank",
        "confined space"
    ],

    "Equipment Maintenance": [
        "during maintenance",
        "equipment maintenance",
        "machine maintenance",
        "pump maintenance",
        "maintenance of the pump"
    ],

    "Lifting Operation": [
        "crane lifting",
        "lifting operation",
        "lifting load",
        "crane operation",
        "rigging"
    ],

    "Hot Work": [
        "hot work",
        "welding",
        "cutting",
        "grinding"
    ],

    "Working at Height": [
        "working at height",
        "climbed a platform",
        "scaffold",
        "scaffolding",
        "ladder",
        "elevated platform"
    ],

    "Excavation": [
        "excavation",
        "excavating",
        "trenching",
        "trench",
        "digging"
    ],

    "Vehicle Operation": [
        "driving",
        "driver",
        "reversing",
        "reverse",
        "truck",
        "forklift",
        "vehicle"
    ],

    "Electrical Maintenance": [
        "electrical maintenance",
        "electrical panel",
        "electrician",
        "electrical work"
    ]
}


# ============================================================
# 4. CLEAN TEXT
# ============================================================

def clean_text(text):

    text = str(text).lower()

    text = re.sub(r"\s+", " ", text)

    return text.strip()


# ============================================================
# 5. GENERIC KEYWORD DETECTOR
# ============================================================

def find_best_match(text, keyword_dictionary):

    matches = []

    for category, keywords in keyword_dictionary.items():

        score = 0

        for keyword in keywords:

            if keyword in text:
                score += 1

        if score > 0:
            matches.append((category, score))

    if not matches:
        return "Not detected", 0

    matches.sort(key=lambda x: x[1], reverse=True)

    return matches[0]


# ============================================================
# 6. ANALYZE REPORT
# ============================================================

def extract_hazard_barrier_activity(report):

    text = clean_text(report)

    hazard, hazard_score = find_best_match(text, HAZARDS)

    barrier, barrier_score = find_best_match(text, BARRIERS)

    activity, activity_score = find_best_match(text, ACTIVITIES)

    return {
        "hazard": hazard,
        "hazard_keyword_matches": hazard_score,

        "barrier_failure": barrier,
        "barrier_keyword_matches": barrier_score,

        "activity": activity,
        "activity_keyword_matches": activity_score
    }


# ============================================================
# 7. TESTING
# ============================================================

if __name__ == "__main__":

    print("=" * 70)
    print("SIH26165 - HAZARD / BARRIER / ACTIVITY EXTRACTOR")
    print("=" * 70)


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


    print("\nTEST RESULTS")
    print("=" * 70)


    for number, report in enumerate(test_reports, start=1):

        result = extract_hazard_barrier_activity(report)

        print("\n" + "-" * 70)

        print(f"Report #{number}")
        print("Report:", report)

        print("\nActivity:")
        print(result["activity"])

        print("Hazard:")
        print(result["hazard"])

        print("Barrier Failure:")
        print(result["barrier_failure"])

        print("\nHazard keyword matches:",
              result["hazard_keyword_matches"])

        print("Barrier keyword matches:",
              result["barrier_keyword_matches"])

        print("Activity keyword matches:",
              result["activity_keyword_matches"])


    print("\n" + "=" * 70)
    print("STEP 4 EXTRACTOR TEST COMPLETED")
    print("=" * 70)