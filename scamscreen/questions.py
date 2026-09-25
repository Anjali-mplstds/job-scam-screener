"""The scam checklist: every check is one typed question answered by Laya in a single forward pass.

Yes/no checks are sent as two-option `choice` questions with neutral keys (A = yes, B = no).
The Laya model card warns that `noul` questions can follow their true/false labels instead of the
input on the English checkpoint, and recommends this neutral-key form. Set `question_mode: noul`
in config.yaml to compare the two on your own data.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Check:
    id: str
    label: str          # short human label shown in the dashboard
    question: str       # instruction text sent to the model
    yes: str            # option A description
    no: str             # option B description
    weight: float       # maximum contribution to the risk score (0-1)
    hard: bool = False  # one confident hit is enough to force a high-risk verdict


CHECKS: list[Check] = [
    Check("upfront_fee", "Asks you to pay money",
          "Does the posting ask the applicant to pay money, such as a registration fee, "
          "training fee, deposit or equipment purchase?",
          "yes, the applicant has to pay money", "no, the applicant does not pay anything",
          0.90, True),
    Check("sensitive_info", "Wants sensitive data up front",
          "Does the posting ask for sensitive personal or financial details, such as ID documents, "
          "bank account numbers or social security numbers, before any hiring decision?",
          "yes, it asks for sensitive personal or financial details",
          "no, it does not ask for sensitive details", 0.80, True),
    Check("money_handling", "Moves money for the 'employer'",
          "Does the job involve receiving payments, cashing cheques, forwarding funds or "
          "reshipping packages on behalf of the employer?",
          "yes, the worker handles money or packages for the employer",
          "no, the job does not involve handling money or packages for the employer", 0.85, True),
    Check("task_scam", "Pay-per-task gig",
          "Is the job about completing simple online tasks, such as liking videos, rating products "
          "or boosting rankings, in return for commissions?",
          "yes, it is a pay-per-task online gig", "no, it is a normal job", 0.80),
    Check("offplatform", "Pushes you off-platform",
          "Does the posting tell applicants to reach the employer only through WhatsApp, Telegram, "
          "text message or a personal email address?",
          "yes, contact is only through a chat app or personal email",
          "no, it uses a company channel or the platform itself", 0.55),
    Check("too_good", "Pay too good to be true",
          "Is the pay unrealistically high for the work described and the experience required?",
          "yes, the pay is unrealistic for this work", "no, the pay is plausible", 0.50),
    Check("vague_company", "Employer is vague",
          "Is the employer's identity unclear, with no company name, website or verifiable details?",
          "yes, the employer cannot be identified", "no, the employer is clearly identified", 0.40),
    Check("no_interview", "Hired with no interview",
          "Does the posting promise a job with no interview, no experience needed or instant hiring?",
          "yes, it promises instant hiring with no interview or experience",
          "no, there is a normal hiring process", 0.40),
    Check("urgency", "Pressure to act fast",
          "Does the posting pressure the applicant to respond immediately, for example with "
          "limited slots or urgent hiring?",
          "yes, it pressures the applicant to act immediately", "no, there is no pressure", 0.30),
]

# A protective check: a clear role at an identifiable employer lowers the score
# (never when a hard check fired).
PROTECT = Check("verifiable_role", "Clear, verifiable role",
                "Does the posting describe concrete responsibilities, requirements and an "
                "identifiable employer?",
                "yes, it describes a concrete role at an identifiable employer",
                "no, the role or employer is unclear", 0.50)

SCAM_TYPE_ID = "scam_type"
SCAM_TYPES = {
    "advance_fee": "asks the applicant to pay fees or deposits up front",
    "task_scam": "pay-per-task gig: like, rate or boost things for commission",
    "money_mule": "moves money or packages for the employer",
    "data_harvest": "collects ID, bank or personal data (identity theft)",
    "genuine": "a normal, genuine job posting",
}

CHECK_BY_ID = {c.id: c for c in CHECKS}
CHECK_BY_ID[PROTECT.id] = PROTECT


def build_questions(mode: str = "ab") -> dict:
    """Build the `questions` dict expected by laya (and Jev-compatible servers)."""
    q: dict = {}
    for c in CHECKS + [PROTECT]:
        if mode == "noul":
            q[c.id] = {"type": "noul", "instructions": c.question}
        else:
            q[c.id] = {"type": "choice", "instructions": c.question,
                       "criteria": {"A": c.yes, "B": c.no}}
    q[SCAM_TYPE_ID] = {"type": "choice",
                       "instructions": "Which kind of job posting is this?",
                       "criteria": dict(SCAM_TYPES)}
    return q


def parse_answers(answers: dict, mode: str = "ab") -> dict:
    """Turn Laya/Jev `answers` into {'p': {check_id: P(yes)}, 'scam_type': {...}}."""
    p: dict[str, float] = {}
    for c in CHECKS + [PROTECT]:
        a = answers.get(c.id)
        if not a:
            continue
        if mode == "noul":
            p[c.id] = float(a["noul"])
        else:
            probs = a["probabilities"]
            p[c.id] = float(probs["A"])
    st = answers.get(SCAM_TYPE_ID)
    scam_type = None
    if st:
        scam_type = {"choice": st["choice"], "probabilities": st["probabilities"]}
    return {"p": p, "scam_type": scam_type}
