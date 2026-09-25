"""Synthetic labelled job postings (fictional companies) for demos and plumbing tests.

Use them to test the pipeline and to record a speed demo. Do NOT quote accuracy measured on them:
the scams are templated, so they are easier than real ones. For a real accuracy number use the public
"Fake Job Postings" (EMSCAD) dataset or your own labelled postings.
"""
from __future__ import annotations

import random

COMPANIES = [("Northbeam Logistics", "northbeam.example"), ("Kestrel Analytics", "kestrel-analytics.example"),
             ("Bluefin Dental Group", "bluefindental.example"), ("Marlowe & Finch LLP", "marlowefinch.example"),
             ("Aldergrove Foods", "aldergrovefoods.example"), ("Tidewater Software", "tidewater.example"),
             ("Ironpine Fabrication", "ironpine.example"), ("Solace Home Care", "solacecare.example"),
             ("Harbourline Retail", "harbourline.example"), ("Pinecrest Schools Trust", "pinecrest.example")]
ROLES = [("Accounts Assistant", "Prepare invoices, reconcile ledgers and support month-end close.",
          "2+ years in bookkeeping; comfortable with spreadsheets."),
         ("Warehouse Associate", "Pick, pack and dispatch orders and keep the stock area safe and tidy.",
          "Ability to lift 20 kg; forklift licence is a plus."),
         ("Software Engineer", "Build and maintain internal web services with a small product team.",
          "Experience with Python or TypeScript and code review."),
         ("Dental Receptionist", "Greet patients, manage appointments and handle insurance paperwork.",
          "Friendly manner; prior front-desk experience preferred."),
         ("Customer Support Agent", "Answer customer emails and chats and log issues in our helpdesk.",
          "Clear written English; one year of support experience."),
         ("Line Cook", "Prepare menu items to specification during evening service.",
          "Food handler card; able to work weekends."),
         ("Data Analyst", "Turn sales data into weekly dashboards for regional managers.",
          "SQL and a BI tool; strong attention to detail."),
         ("Teaching Assistant", "Support classroom teachers with small-group reading sessions.",
          "Enhanced background check required; patience with young learners.")]
CITIES = ["Leeds", "Austin", "Pune", "Toronto", "Manchester", "Denver", "Bengaluru", "Dublin"]
FREEMAIL = ["gmail.com", "yahoo.com", "outlook.com"]


def _legit(rng: random.Random, i: int) -> dict:
    co, dom = rng.choice(COMPANIES)
    role, duties, req = rng.choice(ROLES)
    hard = rng.random() < 0.2
    apply = rng.choice([f"Apply through the careers page at careers.{dom}.",
                        f"Send your CV to hr@{dom} and we will reply within two weeks.",
                        "Apply on this platform; shortlisted candidates get a phone screen then an on-site interview."])
    extra = (" We are hiring urgently for a fast start, so interviews run this week." if hard else "")
    return {"id": f"syn-{i}", "title": role, "company": co, "location": rng.choice(CITIES),
            "salary": rng.choice(["", "£28,000 - £34,000", "$52,000 - $61,000", "₹6,00,000 - ₹8,00,000 per year"]),
            "description": f"{co} is looking for a {role}. {duties} Requirements: {req} "
                           f"Benefits include 25 days holiday, a pension scheme and paid training. "
                           f"Our process has two interview rounds and a reference check.{extra} {apply}",
            "label": 0}


def _scam(rng: random.Random, i: int) -> dict:
    kind = rng.choice(["fee", "task", "mule", "data", "subtle"])
    co, _ = rng.choice(COMPANIES)
    named = rng.random() < 0.5
    mail = f"{rng.choice(['hr.', 'careers.', 'recruit.'])}{co.split()[0].lower()}{rng.randint(10, 99)}@{rng.choice(FREEMAIL)}"
    pay = rng.choice([250, 300, 450, 500, 800])
    fee = rng.choice([49, 75, 99, 120])
    d = {"id": f"syn-{i}", "company": co if named else "", "location": rng.choice(["Remote", "Work from home", "Anywhere"]),
         "label": 1}
    if kind == "fee":
        d.update(title="Data Entry Executive (Work From Home)",
                 description=f"Earn ${pay} per day typing from home. No experience needed, hired instantly. "
                             f"A refundable registration fee of ${fee} is required to receive your starter pack. "
                             f"Apply now, limited slots! Contact us on WhatsApp or email {mail}.")
    elif kind == "task":
        d.update(title="Online Product Rating Assistant",
                 description=f"Complete simple daily tasks: like videos, rate products and boost rankings. "
                             f"Commission per task, make ${pay} a day. Training is on Telegram. "
                             f"No interview required. Add our recruiter on Telegram to get started today.")
    elif kind == "mule":
        d.update(title="Payment Processing Agent",
                 description=f"Receive payments from our customers into your bank account and forward the funds "
                             f"to our suppliers. Keep a 10% commission. We will send a cheque; cash the cheque "
                             f"and transfer the funds the same day. Reply to {mail}.")
    elif kind == "data":
        d.update(title="Remote Administrative Assistant",
                 description=f"Immediate start, ${pay} weekly guaranteed. To set up payroll send a copy of your "
                             f"passport, your social security number and bank account details by email to {mail}. "
                             f"Hiring immediately, respond within 24 hours.")
    else:  # subtle: few of the obvious keywords
        d.update(title="Executive Assistant to the Director",
                 description=f"Our director travels often and needs a trusted assistant to manage errands and "
                             f"purchases on her behalf. Flexible hours, generous weekly allowance. Interested "
                             f"candidates please write to {mail} with your full name, address and date of birth.")
    return d


def generate(n: int = 500, scam_ratio: float = 0.3, seed: int = 7) -> list[dict]:
    rng = random.Random(seed)
    items = [(_scam if rng.random() < scam_ratio else _legit)(rng, i) for i in range(n)]
    return items
