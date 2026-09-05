# -*- coding: utf-8 -*-
"""
Framework detection tests: does the closed set fire on real material, and — the
half that actually matters — does it stay quiet on everything else?

The precision cases are the point of this file. A false positive means we
generate questions on a framework the student's course never taught, which is
strictly worse than generating none. Every "should NOT detect" fixture below is
prose of the kind that genuinely appears in nursing material and contains the
tempting vocabulary ("assessment", "planning", "safety") without teaching the
framework.

No pytest in this project's requirements, so this runs standalone:

    venv/Scripts/python.exe tests/test_nursing_frameworks.py
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from constants.nursing_frameworks import (  # noqa: E402
    FRAMEWORKS,
    build_model_choices,
    detect_frameworks,
    framework_ids,
    get_framework,
)

failures = []


def check(name, condition, detail=""):
    if condition:
        print(f"  PASS  {name}")
    else:
        print(f"  FAIL  {name}  {detail}")
        failures.append(name)


def ids_of(text):
    return [r["id"] for r in detect_frameworks(text)]


# ---------------------------------------------------------------------------
# RECALL — material that genuinely teaches a framework
# ---------------------------------------------------------------------------

print("\nRECALL")

NURSING_PROCESS_DOC = """
    The Nursing Process
    The nursing process is a five-step problem-solving framework. During the
    assessment phase the nurse collects data. The nurse then formulates a
    nursing diagnosis based on that data. In the planning phase the nurse and
    patient set an expected outcome with a target date. The implementation
    phase is where interventions are carried out, and the evaluation phase
    determines whether the expected outcome was met.
"""
check("nursing process — named anchor",
      "nursing_process" in ids_of(NURSING_PROCESS_DOC),
      ids_of(NURSING_PROCESS_DOC))

r = detect_frameworks(NURSING_PROCESS_DOC)[0]
check("nursing process — reports 'named' confidence",
      r["confidence"] == "named", r["confidence"])

# The same content with the framework never named — the inferred path.
ADPIE_UNNAMED = """
    After collecting data in the assessment phase, the nurse writes a nursing
    diagnosis. The planning phase establishes an expected outcome. The
    implementation phase delivers care and the evaluation phase judges results.
"""
check("nursing process — inferred without the anchor",
      "nursing_process" in ids_of(ADPIE_UNNAMED), ids_of(ADPIE_UNNAMED))
check("nursing process — inferred is labelled as such",
      detect_frameworks(ADPIE_UNNAMED)[0]["confidence"] == "inferred")

PRECAUTIONS_DOC = """
    Transmission-Based Precautions
    Standard precautions apply to all patients. Contact precautions require gown
    and gloves. Droplet precautions require a surgical mask within three feet.
    Airborne precautions require an N95 respirator and a negative pressure room.
"""
check("isolation precautions", "isolation_precautions" in ids_of(PRECAUTIONS_DOC),
      ids_of(PRECAUTIONS_DOC))

MASLOW_DOC = """
    Maslow's hierarchy of needs ranks physiological need first, then safety and
    security, then love and belonging, then esteem needs, and finally
    self-actualization.
"""
check("maslow", "maslow" in ids_of(MASLOW_DOC), ids_of(MASLOW_DOC))

ABG_DOC = """
    Arterial blood gas interpretation. A low pH with an elevated PaCO2 indicates
    respiratory acidosis. A high pH with a low PaCO2 indicates respiratory
    alkalosis. A low pH with a low HCO3 is metabolic acidosis, while a high pH
    with elevated bicarbonate is metabolic alkalosis. Compensation may be partial.
"""
check("acid-base", "acid_base" in ids_of(ABG_DOC), ids_of(ABG_DOC))

DELEGATION_DOC = """
    Delegation and scope of practice. The registered nurse may not delegate
    assessment, teaching, or evaluation. Unlicensed assistive personnel may take
    vital signs on a stable patient. The LPN may reinforce teaching already given.
"""
check("delegation", "delegation_scope" in ids_of(DELEGATION_DOC), ids_of(DELEGATION_DOC))

DATA_DOC = """
    Subjective data is what the patient reports, such as pain or nausea.
    Objective data is observable and measurable, such as a blood pressure reading
    or an objective finding on inspection.
"""
check("subjective vs objective", "data_type" in ids_of(DATA_DOC), ids_of(DATA_DOC))


# ---------------------------------------------------------------------------
# PRECISION — material that must NOT trigger a framework
# ---------------------------------------------------------------------------

print("\nPRECISION")

# "Assessment" and "planning" appear constantly in nursing prose. One or two
# generic uses must not be enough.
GENERIC_ASSESSMENT = """
    Respiratory assessment begins with inspection of the chest wall. Auscultate
    all lung fields. Document your assessment findings and report abnormalities
    to the provider. Careful planning of the shift helps the nurse stay ahead.
"""
check("generic 'assessment/planning' prose stays quiet",
      "nursing_process" not in ids_of(GENERIC_ASSESSMENT), ids_of(GENERIC_ASSESSMENT))

PHARM_DOC = """
    Cytochrome P450 metabolism affects half-life. Monitor BUN and creatinine for
    renal clearance. An agonist binds and activates the receptor; an antagonist
    blocks it. Check GFR before administering nephrotoxic drugs.
"""
check("pharmacology doc detects nothing", ids_of(PHARM_DOC) == [], ids_of(PHARM_DOC))

VITALS_DOC = """
    Vital signs measurement and interpretation. Normal adult temperature ranges
    from 36.5 to 37.5 C. Count respirations for a full minute. Blood pressure
    should be taken with the cuff at heart level.
"""
check("vital signs doc detects nothing", ids_of(VITALS_DOC) == [], ids_of(VITALS_DOC))

# "Safety" alone must not pull in Maslow.
SAFETY_DOC = """
    Patient safety in the home environment. Remove throw rugs to prevent falls.
    Ensure adequate lighting. Safety is a priority for the older adult.
"""
check("safety prose does not trigger maslow",
      "maslow" not in ids_of(SAFETY_DOC), ids_of(SAFETY_DOC))

check("empty text detects nothing", detect_frameworks("") == [])
check("whitespace detects nothing", detect_frameworks("   \n  ") == [])


# ---------------------------------------------------------------------------
# REGISTRY INTEGRITY — the table has to stay generatable
# ---------------------------------------------------------------------------

print("\nREGISTRY")

for fid, fw in FRAMEWORKS.items():
    check(f"{fid}: has >= 2 categories", len(fw["categories"]) >= 2)
    check(f"{fid}: has anchors and detect terms",
          bool(fw["anchors"]) and bool(fw["detect_terms"]))
    check(f"{fid}: min_terms is reachable",
          fw["min_terms"] <= len(fw["detect_terms"]),
          f'min_terms={fw["min_terms"]} terms={len(fw["detect_terms"])}')
    # Confusable pairs feed the distractor instructions, so a typo there would
    # silently produce a nonsense prompt.
    for a, b in fw["confusable"]:
        check(f"{fid}: confusable pair is real ({a}/{b})",
              a in fw["categories"] and b in fw["categories"])

check("unknown id returns None", get_framework("not_a_framework") is None)
check("get_framework stamps the id", get_framework("maslow")["id"] == "maslow")

np_default = get_framework("nursing_process")
np_nclex = get_framework("nursing_process", nclex_framing=True)
check("ADPIE is the default framing", np_default["categories"][0] == "Assessment")
check("NCLEX framing swaps in the NCSBN steps",
      np_nclex["categories"][0] == "Recognize Cues", np_nclex["categories"][:1])
check("NCLEX framing does not leak into other frameworks",
      get_framework("maslow", nclex_framing=True)["categories"]
      == FRAMEWORKS["maslow"]["categories"])

choices = build_model_choices()
check("model choice list covers every framework",
      all(fid in choices for fid in framework_ids()))
check("model choice list offers 'none'", "- none:" in choices)


# ---------------------------------------------------------------------------

print("\n" + "=" * 58)
if failures:
    print(f"{len(failures)} FAILED: {', '.join(failures)}")
    sys.exit(1)
print("All framework detection tests passed.")
