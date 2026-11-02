"""
AI Email Triage Simulator
-------------------------
Upload sample emails -> AI labels purpose + urgency, extracts tasks,
and drafts replies for human review.

Stack: Streamlit (UI) + Groq API (LLM) + pandas (tables/export)
"""

import email
import email.policy
import html
import io
import json
import os
import re
from datetime import datetime

import pandas as pd
import streamlit as st
from groq import Groq

# ----------------------------------------------------------------------------
# Page config
# ----------------------------------------------------------------------------
st.set_page_config(
    page_title="Email Triage Simulator",
    page_icon="📬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ----------------------------------------------------------------------------
# Constants
# ----------------------------------------------------------------------------
MODELS = {
    "Llama 3.3 70B (best quality)": "llama-3.3-70b-versatile",
    "Llama 3.1 8B (fastest)": "llama-3.1-8b-instant",
}

PURPOSES = [
    "Action request",
    "Meeting / scheduling",
    "Approval needed",
    "Customer support",
    "Billing / invoice",
    "Sales / marketing",
    "Information / FYI",
    "Personal",
    "Spam / irrelevant",
]

URGENCY_ORDER = {"Critical": 0, "High": 1, "Medium": 2, "Low": 3}
URGENCY_COLORS = {
    "Critical": ("#B4232F", "#FDECEE"),
    "High": ("#B45B0A", "#FEF1E1"),
    "Medium": ("#2F4FBF", "#E9EEFC"),
    "Low": ("#4A6B5D", "#E8F1EC"),
}

TONES = ["Professional", "Friendly", "Concise", "Formal"]

SAMPLE_EMAILS = [
    {
        "from": "Priya Nair <priya.nair@northwind.example>",
        "subject": "Server outage affecting checkout - need update ASAP",
        "body": (
            "Hi team,\n\nCheckout has been failing for about 20 minutes and customers are "
            "complaining on social media. Can someone confirm the root cause and post an "
            "update on the status page within the next 30 minutes? Please also loop in "
            "the on-call engineer and send me a short incident summary by end of day.\n\n"
            "Priya"
        ),
    },
    {
        "from": "Daniel Ortiz <daniel@brightlabs.example>",
        "subject": "Can we move Thursday's sync?",
        "body": (
            "Hey,\n\nSomething came up on my side. Could we move Thursday's product sync "
            "to Friday at 11am or Monday morning? Let me know what works and I'll update "
            "the invite.\n\nThanks,\nDaniel"
        ),
    },
    {
        "from": "Accounts <billing@cloudhost.example>",
        "subject": "Invoice #48213 due on 15 November",
        "body": (
            "Hello,\n\nYour invoice #48213 for $1,240.00 is due on 15 November. "
            "Please pay via the customer portal or reply with your purchase order "
            "number if one is required.\n\nRegards,\nCloudHost Billing"
        ),
    },
    {
        "from": "Growth Weekly <hello@growthweekly.example>",
        "subject": "10 hacks to 10x your pipeline (limited offer inside)",
        "body": (
            "Don't miss our exclusive webinar! Sign up in the next 24 hours and get "
            "50% off our premium playbook. Click here to claim your spot."
        ),
    },
    {
        "from": "Maria Chen <maria.chen@acme.example>",
        "subject": "Budget approval for Q1 design contractor",
        "body": (
            "Hi,\n\nI'd like to bring on a freelance designer for Q1 at roughly $8,500 "
            "total. Could you approve this by Wednesday so I can send the offer? "
            "I've attached the scope in the shared drive. Happy to walk you through it.\n\n"
            "Best,\nMaria"
        ),
    },
    {
        "from": "Sam Patel <sam.patel@customer.example>",
        "subject": "Can't export my report to PDF",
        "body": (
            "Hello support,\n\nEvery time I click Export > PDF on my monthly report I get "
            "an error saying 'Something went wrong'. I tried Chrome and Safari. "
            "I need this for a board meeting on Friday. Can you help?\n\nSam"
        ),
    },
]

# ----------------------------------------------------------------------------
# Styling
# ----------------------------------------------------------------------------
CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Manrope:wght@400;500;600;700;800&display=swap');

html, body, [class*="css"], .stMarkdown, .stTextInput, .stTextArea, .stButton button {
    font-family: 'Manrope', -apple-system, 'Segoe UI', sans-serif;
}
#MainMenu, footer {visibility: hidden;}
.block-container {padding-top: 2rem; padding-bottom: 4rem; max-width: 1180px;}

/* Hero */
.hero {
    background: linear-gradient(135deg, #16233D 0%, #243B6B 100%);
    border-radius: 18px;
    padding: 34px 38px;
    margin-bottom: 22px;
    color: #F3F6FD;
}
.hero h1 {
    font-size: 2.1rem; font-weight: 800; letter-spacing: -0.02em;
    margin: 0 0 8px 0; color: #FFFFFF;
}
.hero p {margin: 0; font-size: 1.02rem; color: #C5D0EA; max-width: 640px; line-height: 1.55;}

/* Metric tiles */
.tile {
    border: 1px solid rgba(120,130,160,0.28);
    border-radius: 14px;
    padding: 16px 18px;
}
.tile .num {font-size: 1.9rem; font-weight: 800; letter-spacing: -0.02em; line-height: 1.1;}
.tile .lbl {font-size: 0.85rem; opacity: 0.7; margin-top: 4px;}

/* Badges */
.badge {
    display: inline-block; padding: 3px 11px; border-radius: 999px;
    font-size: 0.78rem; font-weight: 700; margin-right: 6px;
}
.badge-purpose {background: rgba(120,130,160,0.18);}

/* Email header */
.mail-subject {font-size: 1.08rem; font-weight: 700; margin: 6px 0 2px 0;}
.mail-from {font-size: 0.86rem; opacity: 0.65; margin-bottom: 10px;}
.summary {font-size: 0.95rem; line-height: 1.55; margin: 8px 0 4px 0;}
.section-label {font-size: 0.85rem; font-weight: 700; margin: 14px 0 6px 0; opacity: 0.85;}
.task {
    border-left: 3px solid #2F5BEA; padding: 6px 12px; margin: 6px 0;
    background: rgba(47,91,234,0.07); border-radius: 0 8px 8px 0; font-size: 0.92rem;
}
.task small {opacity: 0.7;}
.empty {text-align:center; padding: 48px 20px; opacity: 0.75;}

/* Primary button */
.stButton button[kind="primary"] {
    background: #2F5BEA; border: none; font-weight: 700; border-radius: 10px;
}
.stButton button[kind="primary"]:hover {background: #2447C4;}
</style>
"""
st.markdown(CSS, unsafe_allow_html=True)


# ----------------------------------------------------------------------------
# Email parsing helpers
# ----------------------------------------------------------------------------
def _clean(text: str) -> str:
    return (text or "").replace("\r\n", "\n").strip()


def _split_headers(block: str) -> dict:
    """Pull optional 'From:' / 'Subject:' lines from the top of a plain-text block."""
    sender, subject, lines = "Unknown sender", "(no subject)", block.split("\n")
    body_start = 0
    for i, line in enumerate(lines[:6]):
        m = re.match(r"^(from|subject|to|date)\s*:\s*(.*)$", line.strip(), re.I)
        if m:
            key, val = m.group(1).lower(), m.group(2).strip()
            if key == "from":
                sender = val or sender
            elif key == "subject":
                subject = val or subject
            body_start = i + 1
        elif line.strip() == "" and body_start:
            body_start = i + 1
            break
        elif not m and body_start == 0:
            break
    body = "\n".join(lines[body_start:]).strip() or block
    return {"from": sender, "subject": subject, "body": body}


def parse_upload(uploaded) -> list[dict]:
    """Turn an uploaded file into a list of {from, subject, body} dicts."""
    name = uploaded.name.lower()
    raw = uploaded.getvalue()

    if name.endswith(".eml"):
        msg = email.message_from_bytes(raw, policy=email.policy.default)
        body_part = msg.get_body(preferencelist=("plain", "html"))
        body = body_part.get_content() if body_part else ""
        if body_part and body_part.get_content_type() == "text/html":
            body = re.sub(r"<[^>]+>", " ", body)
        return [{
            "from": str(msg.get("From", "Unknown sender")),
            "subject": str(msg.get("Subject", "(no subject)")),
            "body": _clean(body),
        }]

    if name.endswith(".csv"):
        df = pd.read_csv(io.BytesIO(raw)).fillna("")
        cols = {c.lower().strip(): c for c in df.columns}
        pick = lambda *opts: next((cols[o] for o in opts if o in cols), None)
        c_from = pick("from", "sender", "email")
        c_subj = pick("subject", "title")
        c_body = pick("body", "message", "content", "text")
        if not c_body:
            raise ValueError("CSV needs a 'body' column (optional: 'from', 'subject').")
        return [{
            "from": str(r[c_from]) if c_from else "Unknown sender",
            "subject": str(r[c_subj]) if c_subj else "(no subject)",
            "body": _clean(str(r[c_body])),
        } for _, r in df.iterrows() if str(r[c_body]).strip()]

    if name.endswith(".json"):
        data = json.loads(raw.decode("utf-8", errors="ignore"))
        if isinstance(data, dict):
            data = data.get("emails", [data])
        return [{
            "from": str(d.get("from", d.get("sender", "Unknown sender"))),
            "subject": str(d.get("subject", "(no subject)")),
            "body": _clean(str(d.get("body", d.get("message", "")))),
        } for d in data if str(d.get("body", d.get("message", ""))).strip()]

    # .txt / .md  -> multiple emails separated by a line of --- or ===
    text = _clean(raw.decode("utf-8", errors="ignore"))
    blocks = [b.strip() for b in re.split(r"\n\s*(?:-{3,}|={3,})\s*\n", text) if b.strip()]
    return [_split_headers(b) for b in blocks]


# ----------------------------------------------------------------------------
# Groq calls
# ----------------------------------------------------------------------------
def get_api_key() -> str:
    try:
        if "GROQ_API_KEY" in st.secrets:
            return st.secrets["GROQ_API_KEY"]
    except Exception:
        pass
    return os.getenv("GROQ_API_KEY", "")


SYSTEM_PROMPT = f"""You are an executive email triage assistant. Analyse ONE email and
respond with a single JSON object and nothing else.

Schema:
{{
  "purpose": one of {PURPOSES},
  "urgency": one of ["Critical", "High", "Medium", "Low"],
  "urgency_reason": short sentence explaining the urgency level,
  "summary": 1-2 sentence plain-language summary,
  "tasks": [{{"task": "imperative action item", "owner": "who should do it, or 'Me'", "due": "deadline from the email or 'Not specified'"}}],
  "needs_reply": true or false,
  "draft_reply": "a ready-to-send reply (empty string if no reply is needed)",
  "confidence": number between 0 and 1
}}

Rules:
- Critical = outage, legal/security risk, or deadline within hours.
- High = deadline within ~2 days or a blocked colleague/customer.
- Medium = needs action but no pressing deadline. Low = FYI, marketing, spam.
- Only list tasks that are really in the email. Use an empty list if there are none.
- Never invent facts, names, prices or dates in the draft. If information is missing, say you'll follow up.
- Newsletters, spam and automated notices do not need a reply.
"""


def call_groq(client: Groq, model: str, messages: list, json_mode: bool = True) -> str:
    kwargs = dict(model=model, messages=messages, temperature=0.2, max_tokens=1200)
    if json_mode:
        kwargs["response_format"] = {"type": "json_object"}
    resp = client.chat.completions.create(**kwargs)
    return resp.choices[0].message.content


def normalise(result: dict) -> dict:
    """Make sure the model output has all the fields we rely on."""
    purpose = result.get("purpose", "Information / FYI")
    if purpose not in PURPOSES:
        purpose = "Information / FYI"
    urgency = str(result.get("urgency", "Medium")).title()
    if urgency not in URGENCY_ORDER:
        urgency = "Medium"
    tasks = []
    for t in result.get("tasks", []) or []:
        if isinstance(t, dict) and t.get("task"):
            tasks.append({
                "task": str(t["task"]),
                "owner": str(t.get("owner", "Me")),
                "due": str(t.get("due", "Not specified")),
            })
    try:
        conf = float(result.get("confidence", 0.8))
    except (TypeError, ValueError):
        conf = 0.8
    return {
        "purpose": purpose,
        "urgency": urgency,
        "urgency_reason": str(result.get("urgency_reason", "")),
        "summary": str(result.get("summary", "")),
        "tasks": tasks,
        "needs_reply": bool(result.get("needs_reply", False)),
        "draft_reply": str(result.get("draft_reply", "") or ""),
        "confidence": max(0.0, min(1.0, conf)),
    }


def analyse_email(client: Groq, model: str, mail: dict, tone: str, signoff: str) -> dict:
    user_msg = (
        f"Reply tone: {tone}\nSign replies as: {signoff or 'the recipient'}\n\n"
        f"From: {mail['from']}\nSubject: {mail['subject']}\n\n"
        f"{mail['body'][:6000]}"
    )
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_msg},
    ]
    last_err = None
    for _ in range(2):  # one retry on malformed output
        try:
            return normalise(json.loads(call_groq(client, model, messages)))
        except Exception as e:  # noqa: BLE001
            last_err = e
    raise RuntimeError(str(last_err))


def regenerate_reply(idx: int):
    """Callback: write a fresh draft for one email using the current tone."""
    mail = st.session_state.emails[idx]
    key = get_api_key() or st.session_state.get("manual_key", "")
    if not key:
        st.session_state["toast"] = "Add your Groq API key first."
        return
    client = Groq(api_key=key)
    tone = st.session_state.get("tone", "Professional")
    signoff = st.session_state.get("signoff", "")
    prompt = (
        f"Write a {tone.lower()} reply to this email. Do not invent facts. "
        f"Sign off as {signoff or 'the recipient'}. Return only the reply text.\n\n"
        f"From: {mail['from']}\nSubject: {mail['subject']}\n\n{mail['body'][:6000]}"
    )
    try:
        text = call_groq(
            client, st.session_state.get("model_id", "llama-3.3-70b-versatile"),
            [{"role": "user", "content": prompt}], json_mode=False,
        )
        st.session_state[f"draft_{idx}"] = text.strip()
        st.session_state.results[idx]["draft_reply"] = text.strip()
        st.session_state.results[idx]["needs_reply"] = True
    except Exception as e:  # noqa: BLE001
        st.session_state["toast"] = f"Could not regenerate: {e}"


# ----------------------------------------------------------------------------
# Session state
# ----------------------------------------------------------------------------
st.session_state.setdefault("emails", [])
st.session_state.setdefault("results", {})
st.session_state.setdefault("errors", {})

# ----------------------------------------------------------------------------
# Sidebar
# ----------------------------------------------------------------------------
with st.sidebar:
    st.markdown("### Setup")

    secret_key = get_api_key()
    if secret_key:
        st.success("Groq API key loaded", icon="🔑")
        api_key = secret_key
    else:
        api_key = st.text_input(
            "Groq API key", type="password", key="manual_key",
            help="Get a free key at console.groq.com. On Streamlit Cloud, store it in Secrets as GROQ_API_KEY.",
        )

    model_label = st.selectbox("Model", list(MODELS.keys()))
    st.session_state["model_id"] = MODELS[model_label]

    st.markdown("### Reply style")
    tone = st.selectbox("Tone", TONES, key="tone")
    signoff = st.text_input("Sign replies as", placeholder="Your name", key="signoff")

    st.markdown("### Emails")
    uploads = st.file_uploader(
        "Upload emails",
        type=["txt", "md", "eml", "csv", "json"],
        accept_multiple_files=True,
        help=(
            ".eml = one email. .txt = several emails separated by a line of ---. "
            ".csv / .json = columns or keys: from, subject, body."
        ),
    )

    c1, c2 = st.columns(2)
    if c1.button("Load files", use_container_width=True, disabled=not uploads):
        loaded, problems = [], []
        for f in uploads:
            try:
                loaded.extend(parse_upload(f))
            except Exception as e:  # noqa: BLE001
                problems.append(f"{f.name}: {e}")
        if loaded:
            st.session_state.emails = loaded
            st.session_state.results, st.session_state.errors = {}, {}
            for k in [k for k in st.session_state if str(k).startswith("draft_")]:
                del st.session_state[k]
        for p in problems:
            st.error(p)
        if not loaded and not problems:
            st.warning("No emails found in the uploaded files.")

    if c2.button("Use samples", use_container_width=True):
        st.session_state.emails = [dict(m) for m in SAMPLE_EMAILS]
        st.session_state.results, st.session_state.errors = {}, {}
        for k in [k for k in st.session_state if str(k).startswith("draft_")]:
            del st.session_state[k]

    with st.expander("File format examples"):
        st.code(
            "From: Ana <ana@example.com>\nSubject: Quick question\n\n"
            "Can you send the deck by Friday?\n\n---\n\n"
            "From: Ops <ops@example.com>\nSubject: Maintenance window\n\n"
            "Systems will be down Sunday 2-4am.",
            language="text",
        )

# ----------------------------------------------------------------------------
# Header
# ----------------------------------------------------------------------------
st.markdown(
    """
    <div class="hero">
        <h1>Email Triage Simulator</h1>
        <p>Load a batch of emails and get each one labelled by purpose and urgency,
        with action items pulled out and a reply drafted for you to review before anything is sent.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

toast_msg = st.session_state.pop("toast", None)
if toast_msg:
    st.toast(toast_msg)

emails = st.session_state.emails

# ----------------------------------------------------------------------------
# Empty state
# ----------------------------------------------------------------------------
if not emails:
    st.markdown(
        """
        <div class="empty">
            <h3>No emails loaded yet</h3>
            <p>Upload files from the sidebar, or choose <b>Use samples</b> to try the demo.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.stop()

# ----------------------------------------------------------------------------
# Run analysis
# ----------------------------------------------------------------------------
pending = [i for i in range(len(emails)) if i not in st.session_state.results]

top_l, top_r = st.columns([3, 1.4])
top_l.markdown(
    f"**{len(emails)} emails loaded** &nbsp;·&nbsp; {len(st.session_state.results)} analysed"
)
run_label = "Analyse emails" if pending else "Re-run analysis"
if top_r.button(run_label, type="primary", use_container_width=True):
    if not api_key:
        st.error("Add your Groq API key in the sidebar to continue.")
        st.stop()
    client = Groq(api_key=api_key)
    targets = pending if pending else list(range(len(emails)))
    if not pending:
        st.session_state.results, st.session_state.errors = {}, {}
        for k in [k for k in st.session_state if str(k).startswith("draft_")]:
            del st.session_state[k]
    bar = st.progress(0.0, text="Starting…")
    for n, i in enumerate(targets, start=1):
        bar.progress((n - 1) / len(targets), text=f"Reading email {n} of {len(targets)}…")
        try:
            res = analyse_email(client, st.session_state["model_id"], emails[i], tone, signoff)
            st.session_state.results[i] = res
            st.session_state[f"draft_{i}"] = res["draft_reply"]
            st.session_state.errors.pop(i, None)
        except Exception as e:  # noqa: BLE001
            st.session_state.errors[i] = str(e)
    bar.empty()
    st.rerun()

results = st.session_state.results

if st.session_state.errors:
    with st.expander(f"{len(st.session_state.errors)} emails could not be analysed", expanded=False):
        for i, err in st.session_state.errors.items():
            st.write(f"**{emails[i]['subject']}** — {err}")
        st.caption("Check your API key and rate limits, then select Analyse emails to retry.")

if not results:
    st.info("Select **Analyse emails** to start triaging.")
    st.stop()

# ----------------------------------------------------------------------------
# Metrics
# ----------------------------------------------------------------------------
n_urgent = sum(1 for r in results.values() if r["urgency"] in ("Critical", "High"))
n_tasks = sum(len(r["tasks"]) for r in results.values())
n_reply = sum(1 for r in results.values() if r["needs_reply"])
n_noise = sum(1 for r in results.values() if r["purpose"] == "Spam / irrelevant")

m1, m2, m3, m4 = st.columns(4)
for col, num, lbl in [
    (m1, len(results), "Emails triaged"),
    (m2, n_urgent, "Critical or high urgency"),
    (m3, n_tasks, "Action items found"),
    (m4, n_reply, "Replies drafted"),
]:
    col.markdown(
        f'<div class="tile"><div class="num">{num}</div><div class="lbl">{lbl}</div></div>',
        unsafe_allow_html=True,
    )
st.write("")

# ----------------------------------------------------------------------------
# Tabs
# ----------------------------------------------------------------------------
tab_inbox, tab_tasks, tab_export = st.tabs(["Inbox", "Action items", "Export"])

# ---- Inbox -----------------------------------------------------------------
with tab_inbox:
    f1, f2, f3, f4 = st.columns([1.5, 1.5, 1.6, 1.2])
    sel_urg = f1.multiselect("Urgency", list(URGENCY_ORDER), placeholder="All urgency levels")
    sel_pur = f2.multiselect("Purpose", PURPOSES, placeholder="All purposes")
    query = f3.text_input("Search", placeholder="Search sender, subject or summary")
    only_reply = f4.toggle("Needs reply only", value=False)
    hide_noise = st.checkbox("Hide spam and irrelevant mail", value=False)

    def passes(i: int, r: dict) -> bool:
        m = emails[i]
        if sel_urg and r["urgency"] not in sel_urg:
            return False
        if sel_pur and r["purpose"] not in sel_pur:
            return False
        if only_reply and not r["needs_reply"]:
            return False
        if hide_noise and r["purpose"] == "Spam / irrelevant":
            return False
        if query:
            hay = f"{m['from']} {m['subject']} {r['summary']}".lower()
            if query.lower() not in hay:
                return False
        return True

    visible = sorted(
        [i for i, r in results.items() if passes(i, r)],
        key=lambda i: (URGENCY_ORDER[results[i]["urgency"]], i),
    )
    st.caption(f"Showing {len(visible)} of {len(results)} emails, most urgent first")

    if not visible:
        st.markdown(
            '<div class="empty"><b>Nothing matches these filters.</b><br>Clear a filter to see more emails.</div>',
            unsafe_allow_html=True,
        )

    for i in visible:
        m, r = emails[i], results[i]
        fg, bg = URGENCY_COLORS[r["urgency"]]
        with st.container(border=True):
            st.markdown(
                f'<span class="badge" style="color:{fg};background:{bg};">{r["urgency"]}</span>'
                f'<span class="badge badge-purpose">{html.escape(r["purpose"])}</span>'
                f'<div class="mail-subject">{html.escape(m["subject"])}</div>'
                f'<div class="mail-from">{html.escape(m["from"])}</div>'
                f'<div class="summary">{html.escape(r["summary"])}</div>',
                unsafe_allow_html=True,
            )
            if r["urgency_reason"]:
                st.caption(f"Why {r['urgency'].lower()}: {r['urgency_reason']}")

            if r["tasks"]:
                st.markdown('<div class="section-label">Action items</div>', unsafe_allow_html=True)
                for t in r["tasks"]:
                    st.markdown(
                        f'<div class="task">{html.escape(t["task"])}<br>'
                        f'<small>Owner: {html.escape(t["owner"])} &nbsp;·&nbsp; Due: {html.escape(t["due"])}</small></div>',
                        unsafe_allow_html=True,
                    )

            with st.expander("Original email"):
                st.text(m["body"])

            if r["needs_reply"] or st.session_state.get(f"draft_{i}"):
                st.markdown('<div class="section-label">Draft reply</div>', unsafe_allow_html=True)
                st.text_area(
                    "Edit the draft", key=f"draft_{i}", height=190, label_visibility="collapsed",
                )
                b1, b2, _ = st.columns([1.2, 1.2, 3])
                b1.checkbox("Approved", key=f"approved_{i}")
                b2.button("Rewrite reply", key=f"regen_{i}", on_click=regenerate_reply, args=(i,))
            else:
                st.caption("No reply needed for this email.")
                st.button("Draft a reply anyway", key=f"regen_{i}", on_click=regenerate_reply, args=(i,))

# ---- Action items ----------------------------------------------------------
with tab_tasks:
    rows = []
    for i in sorted(results, key=lambda i: (URGENCY_ORDER[results[i]["urgency"]], i)):
        for t in results[i]["tasks"]:
            rows.append({
                "Urgency": results[i]["urgency"],
                "Task": t["task"],
                "Owner": t["owner"],
                "Due": t["due"],
                "From email": emails[i]["subject"],
            })
    if rows:
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
    else:
        st.markdown(
            '<div class="empty"><b>No action items found.</b><br>Emails with clear requests will show tasks here.</div>',
            unsafe_allow_html=True,
        )

# ---- Export ----------------------------------------------------------------
with tab_export:
    export_rows, export_json = [], []
    for i in sorted(results):
        r, m = results[i], emails[i]
        draft = st.session_state.get(f"draft_{i}", r["draft_reply"])
        approved = bool(st.session_state.get(f"approved_{i}", False))
        export_rows.append({
            "from": m["from"], "subject": m["subject"], "purpose": r["purpose"],
            "urgency": r["urgency"], "summary": r["summary"],
            "tasks": " | ".join(f'{t["task"]} (owner: {t["owner"]}, due: {t["due"]})' for t in r["tasks"]),
            "needs_reply": r["needs_reply"], "draft_reply": draft, "approved": approved,
            "confidence": round(r["confidence"], 2),
        })
        export_json.append({**m, **r, "draft_reply": draft, "approved": approved})

    st.markdown("Download your triaged results to share or import elsewhere.")
    stamp = datetime.now().strftime("%Y%m%d_%H%M")
    d1, d2, _ = st.columns([1, 1, 2])
    d1.download_button(
        "Download CSV", pd.DataFrame(export_rows).to_csv(index=False).encode("utf-8"),
        file_name=f"email_triage_{stamp}.csv", mime="text/csv", use_container_width=True,
    )
    d2.download_button(
        "Download JSON", json.dumps(export_json, indent=2).encode("utf-8"),
        file_name=f"email_triage_{stamp}.json", mime="application/json", use_container_width=True,
    )
    st.dataframe(pd.DataFrame(export_rows), use_container_width=True, hide_index=True)

st.caption("Drafts are suggestions only. Review every reply before sending.")
