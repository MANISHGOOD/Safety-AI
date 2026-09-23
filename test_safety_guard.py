import json
import sys
import urllib.request

API = "http://127.0.0.1:8080/api/analyze"


def analyze(text):
    body = json.dumps({"report_text": text}).encode("utf-8")
    req = urllib.request.Request(API, data=body, headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req) as resp:
        return json.loads(resp.read().decode("utf-8"))


tests = [
    (
        "TEST A: ML LOW + Toxic Atmosphere + Confined Space",
        "discussion about gas testing in the confined space was held in the toolbox talk",
    ),
    (
        "TEST B: ML MEDIUM + critical hazard (Flammable vapors during hot work)",
        "a near miss occurred when a welder observed residual flammable vapors near the work zone while performing hot work",
    ),
    (
        "TEST C: ML HIGH + critical hazard + barrier failure (confined space entry, no gas test)",
        "A technician entered a vessel without checking for toxic gases.",
    ),
    (
        "TEST D: LOW report with no critical indicators (controls present)",
        "A worker used proper PPE while performing routine housekeeping.",
    ),
    (
        "TEST E: Unclear report with no detected hazard",
        "saw something at the plant today",
    ),
    (
        "TEST F: ML HIGH with no critical evidence",
        "an extensive audit of the facility revealed many violations and hazardous findings across all areas",
    ),
]

results = []
for label, text in tests:
    d = analyze(text)
    results.append((label, text, d))
    print()
    print("=" * 72)
    print(label)
    print("  Text:", text)
    print(f"  ML Prediction:   {d['ml_prediction']} ({d['ml_confidence']}%)")
    print(f"  Final Risk:      {d['risk_level']}")
    print(f"  Override:        {d['risk_override']}  safety_review={d['safety_review_required']}")
    print(f"  Evidence:        {d['critical_evidence']}  score={d['evidence_score']}")
    print(f"  Hazard/Rule/Barrier: {d['hazard']} | {d['life_saving_rule']} | {d['barrier_failure']}")
    print(f"  Explanation:     {d['explanation']}")
    print(f"  Recommended:     {d['recommended_action']}")
    probs = d["class_probabilities"]
    ps = probs.get("HIGH", 0) + probs.get("MEDIUM", 0) + probs.get("LOW", 0)
    print(f"  Prob sum:        {ps:.2f}%")

print()
print("=" * 72)
print("PASS / FAIL SUMMARY")
print("=" * 72)
ok_all = True

_, _, A = results[0]
ok = A["risk_override"] and A["risk_level"] != "LOW" and A["safety_review_required"]
print(f"TEST A (guard escalates LOW + critical): {'PASS' if ok else 'FAIL'} -> final={A['risk_level']}, override={A['risk_override']}")
ok_all = ok_all and ok

_, _, B = results[1]
ok = (B["risk_override"] and B["safety_review_required"]) or B["risk_level"] in ("MEDIUM", "HIGH")
print(f"TEST B (guard flags MEDIUM + critical):   {'PASS' if ok else 'FAIL'} -> final={B['risk_level']}, override={B['risk_override']}")
ok_all = ok_all and ok

_, _, C = results[2]
ok = (
    C["risk_level"] == "HIGH"
    and not C["risk_override"]
    and C["safety_review_required"]
    and bool(C.get("review_reason"))
)
print(f"TEST C (ML HIGH stays HIGH but review required): {'PASS' if ok else 'FAIL'} -> final={C['risk_level']}, override={C['risk_override']}, review={C['safety_review_required']}")
ok_all = ok_all and ok

_, _, D = results[3]
ok = D["risk_level"] == "LOW" and not D["risk_override"]
print(f"TEST D (LOW safe stays LOW):              {'PASS' if ok else 'FAIL'} -> final={D['risk_level']}, override={D['risk_override']}")
ok_all = ok_all and ok

_, _, E = results[4]
ok = E["risk_level"] == "LOW" and not E["risk_override"]
print(f"TEST E (unclear stays LOW):               {'PASS' if ok else 'FAIL'} -> final={E['risk_level']}, override={E['risk_override']}")
ok_all = ok_all and ok

_, _, F = results[5]
ok_f = (
    F["risk_level"] == "HIGH"
    and not F["risk_override"]
    and not F["safety_review_required"]
    and not F.get("review_reason")
    and F["evidence_score"] == 0
)
print(f"TEST F (ML HIGH, no evidence, no review):  {'PASS' if ok_f else 'FAIL'} -> final={F['risk_level']}, override={F['risk_override']}, review={F['safety_review_required']}")
ok_all = ok_all and ok_f

# Extra check: no false "controls in place" claims in overridden cases
for label, _, d in results:
    if "Standard safety protocols appear to be in place" in d["explanation"]:
        print(f"FAIL: {label} contains forbidden controls claim")
        ok_all = False
    if d["risk_override"] and "controls were in use" in d["explanation"].lower():
        print(f"FAIL: {label} claims controls in use despite override")
        ok_all = False

print()
print("ALL 6 TESTS PASSED" if ok_all else "SOME TESTS FAILED")
sys.exit(0 if ok_all else 1)