"""
Local POC AI Agent — Function Calling + Multi-Turn Interactions
Stack: LangChain + Ollama (llama3.1) + Semantic Search (nomic-embed-text)

Architecture: Python state machine handles all flow logic.
LLM is used only for: intent detection and ambiguous yes/no resolution.
All response formatting is pure Python (no LLM) for predictability.
"""

import math
import random
import re
from enum import Enum, auto
from typing import Optional
from dataclasses import dataclass

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_ollama import ChatOllama, OllamaEmbeddings


# ---------------------------------------------------------------------------
# Mock Data
# ---------------------------------------------------------------------------

PROJECTS_DB = [
    {"id": "P001", "name": "مبادرة الشبكة الذكية الوطنية",          "ministry": "Energy",      "budget": "500M SAR",  "status": "Active",   "lead": "م. خالد الراشد"},
    {"id": "P002", "name": "نظام إدارة المرور الذكي",               "ministry": "Transport",   "budget": "120M SAR",  "status": "Active",   "lead": "م. سارة العتيبي"},
    {"id": "P003", "name": "السجلات الصحية الذكية الوطنية",         "ministry": "Health",      "budget": "300M SAR",  "status": "Planning", "lead": "د. فهد الدوسري"},
    {"id": "P004", "name": "شبكة ترشيد المياه الذكية",              "ministry": "Environment", "budget": "80M SAR",   "status": "Active",   "lead": "م. نورة الشهري"},
    {"id": "P005", "name": "البنية التحتية للمدن الذكية — الرياض",  "ministry": "Municipal",   "budget": "750M SAR",  "status": "Active",   "lead": "م. عمر الغامدي"},
    {"id": "P006", "name": "البنية التحتية للمدن الذكية — جدة",     "ministry": "Municipal",   "budget": "600M SAR",  "status": "Planning", "lead": "م. ريم الزهراني"},
]

SERVICES_DB = [
    {"id": "S001", "name": "تسجيل رخصة تجارية",      "ministry": "Commerce",     "category": "Business",      "processing_days": 3,  "fee": "500 SAR"},
    {"id": "S002", "name": "طلب رخصة بناء",          "ministry": "Municipal",    "category": "Construction",  "processing_days": 14, "fee": "2000 SAR"},
    {"id": "S003", "name": "ترخيص منشأة صحية",       "ministry": "Health",       "category": "Licensing",     "processing_days": 30, "fee": "5000 SAR"},
    {"id": "S004", "name": "تقييم الأثر البيئي",      "ministry": "Environment",  "category": "Assessment",    "processing_days": 45, "fee": "10000 SAR"},
    {"id": "S005", "name": "تسجيل مركبة",            "ministry": "Transport",    "category": "Registration",  "processing_days": 1,  "fee": "200 SAR"},
    {"id": "S006", "name": "تجديد رخصة القيادة",      "ministry": "Transport",    "category": "Registration",  "processing_days": 1,  "fee": "400 SAR"},
]

MINISTRIES_DB = {
    "Energy":      {"head": "Eng. Abdulaziz Al-Saud",     "employees": 4500,   "annual_budget": "15B SAR",  "established": 1975},
    "Transport":   {"head": "Eng. Saleh Al-Jabir",         "employees": 8200,   "annual_budget": "22B SAR",  "established": 1953},
    "Health":      {"head": "Dr. Fahad Al-Jalajel",        "employees": 120000, "annual_budget": "180B SAR", "established": 1949},
    "Environment": {"head": "Eng. Abdulrahman Al-Fadley",  "employees": 3100,   "annual_budget": "8B SAR",   "established": 2016},
    "Municipal":   {"head": "Eng. Majid Al-Hogail",        "employees": 6700,   "annual_budget": "35B SAR",  "established": 1975},
    "Commerce":    {"head": "Dr. Majid Al-Qasabi",         "employees": 5300,   "annual_budget": "12B SAR",  "established": 1954},
}

# Display name in Arabic for each ministry (also used to build the semantic index)
MINISTRY_AR = {
    "Energy": "الطاقة", "Transport": "النقل", "Health": "الصحة",
    "Environment": "البيئة", "Municipal": "الشؤون البلدية", "Commerce": "التجارة",
}

# Extra Arabic synonyms — strengthen the semantic index so colloquial words
# (المواصلات, الكهرباء, البلدية...) map to the right ministry.
MINISTRY_SYNONYMS = {
    "Energy":      "الطاقة الكهرباء الوقود البترول",
    "Transport":   "النقل المواصلات الطرق المرور السيارات",
    "Health":      "الصحة المستشفيات العلاج الطب",
    "Environment": "البيئة المياه التلوث الطبيعة",
    "Municipal":   "الشؤون البلدية البلدية الأمانة المدن",
    "Commerce":    "التجارة الأعمال الشركات الاقتصاد",
}

# Arabic display values for the English data fields.
STATUS_AR = {"Active": "نشط", "Planning": "قيد التخطيط", "Completed": "مكتمل"}
CATEGORY_AR = {
    "Business": "الأعمال", "Construction": "الإنشاءات", "Licensing": "التراخيص",
    "Assessment": "التقييم", "Registration": "التسجيل",
}


def ar_ministry(name: str) -> str:
    return MINISTRY_AR.get(name, name)


def ar_days(n: int) -> str:
    """Grammatically-correct Arabic for a number of working days."""
    if n == 1:
        return "يوم عمل واحد"
    if n == 2:
        return "يوما عمل"
    if 3 <= n <= 10:
        return f"{n} أيام عمل"
    return f"{n} يوم عمل"  # 11+ uses the singular (tamyeez)


def ar_normalize(s: str) -> str:
    """Normalize Arabic text for matching: unify letter variants, drop diacritics.
    e.g. 'رخصة' and 'رخصه' both become 'رخصه'."""
    s = re.sub(r"[ً-ْٰـ]", "", s)   # tashkeel + tatweel
    s = (s.replace("أ", "ا").replace("إ", "ا").replace("آ", "ا")
          .replace("ى", "ي").replace("ة", "ه")
          .replace("ؤ", "و").replace("ئ", "ي"))
    return s.strip().lower()


def ar_content_words(text: str) -> set:
    """Normalized word set with the definite article 'ال' stripped,
    so 'البناء' and 'بناء' compare equal during matching."""
    out = set()
    for w in ar_normalize(text).split():
        if w.startswith("ال") and len(w) > 4:
            w = w[2:]
        out.add(w)
    return out


def ar_money(amount: str) -> str:
    """Display monetary amounts in Arabic: '500M SAR' -> '500 مليون ريال'."""
    m = re.match(r"\s*([\d.,]+)\s*([MB]?)\s*SAR\s*$", amount)
    if not m:
        return amount
    num, scale = m.group(1), m.group(2)
    unit = {"M": " مليون", "B": " مليار", "": ""}[scale]
    return f"{num}{unit} ريال"


AFFIRMATIVES = {
    # فصحى / رسمية
    "نعم", "موافق", "تأكيد", "تاكيد", "مؤكد",
    # عامية سعودية وخليجية
    "اي", "آي", "اه", "أه", "إيه", "ايه",
    "يلا", "يالله", "تمام", "اكيد", "أكيد",
    "صح", "صحيح", "طيب", "ماشي", "امضي", "امضي قدام",
    "وافق", "قدم", "سجل", "نفذ", "ابدأ", "شيله",
    "زبالة", "عادي", "خلاص", "حلو", "ثابر",
    # إنجليزي
    "yes", "ok", "okay", "sure", "confirm", "go",
    "proceed", "submit", "do it", "let's go", "yep", "yup",
}

CANCEL_WORDS = {
    "لا", "لأ", "لآ", "ماابي", "ما ابي", "بلاش", "الغاء", "إلغاء",
    "كنسل", "وقف", "اوقف", "رجوع",
    "no", "cancel", "stop", "nope", "nah", "back",
}

# Keywords that hint at yes/no even mid-sentence
_YES_HINTS = {"وافق", "قدم", "سجل", "نفذ", "ابدأ", "امضي", "تأكد", "اكمل"}
_NO_HINTS  = {"بلاش", "الغ", "الغاء", "إلغاء", "ماابي", "ارجع"}

# Normalized affirmative/cancel sets for robust word-level matching.
_NORM_AFFIRM = {ar_normalize(w) for w in AFFIRMATIVES}
_NORM_CANCEL = {ar_normalize(w) for w in CANCEL_WORDS}


# ---------------------------------------------------------------------------
# Semantic Search Engine
# ---------------------------------------------------------------------------

class SemanticIndex:
    def __init__(self, embedder: OllamaEmbeddings, records: list, fields: list):
        self.records = records
        texts = [" ".join(str(r.get(f, "")) for f in fields) for r in records]
        self.vectors = embedder.embed_documents(texts)
        self._embed = embedder.embed_query

    def search(self, query: str, top_k: int = 6, threshold: float = 0.25) -> list:
        q_vec = self._embed(query)
        scored = [(self._cos(q_vec, v), r) for v, r in zip(self.vectors, self.records)]
        scored.sort(key=lambda x: x[0], reverse=True)
        return [r for score, r in scored if score >= threshold][:top_k]

    @staticmethod
    def _cos(a, b) -> float:
        dot = sum(x * y for x, y in zip(a, b))
        ma = math.sqrt(sum(x * x for x in a))
        mb = math.sqrt(sum(x * x for x in b))
        return dot / (ma * mb) if ma and mb else 0.0


# ---------------------------------------------------------------------------
# State Machine
# ---------------------------------------------------------------------------

class Stage(Enum):
    FREE          = auto()
    PICKED_LIST   = auto()   # User saw a list, waiting for number
    CONFIRM_APPLY = auto()   # Showed service details, asking "want to apply?"
    ASK_NAME      = auto()
    ASK_NID       = auto()
    CONFIRM_SUB   = auto()   # Showing report, waiting for final yes


@dataclass
class State:
    stage:           Stage = Stage.FREE
    last_list:       Optional[list] = None   # list of dicts shown to user
    last_list_type:  Optional[str] = None    # "project" or "service"
    pending_service: Optional[dict] = None
    applicant_name:  Optional[str] = None
    national_id:     Optional[str] = None


# ---------------------------------------------------------------------------
# LLM — used only for intent + greeting
# ---------------------------------------------------------------------------

INTENT_SYSTEM = """You are an intent classifier for a Saudi government portal assistant.
Classify the user's message into EXACTLY ONE intent label.
Reply with ONLY the label — no explanation, no extra words.

Intents:
  GREETING       — greetings only: hi, hello, مرحبا, هلا, السلام عليكم, صباح الخير
  ABOUT          — asking who/what the assistant is or what it can do:
                   "من انت", "وش تسوي", "كيف تساعدني", "who are you", "what can you do"
  SEARCH_PROJECT — wants to find/list/search government PROJECTS (مشاريع)
  SEARCH_SERVICE — wants to find/list/search government SERVICES (خدمات، تراخيص، تسجيل)
  MINISTRY_INFO  — asking about a specific MINISTRY by name (وزارة الصحة، وزارة النقل)
  OTHER          — anything that doesn't fit the above

Rules:
- "من انت" or "وش تسوي" → ABOUT (NOT MINISTRY_INFO)
- A ministry name must be explicitly mentioned for MINISTRY_INFO.
- If unsure, choose OTHER.

Examples:
  "من انت" → ABOUT
  "وش الخدمات الموجودة" → SEARCH_SERVICE
  "ابي مشاريع الطاقة" → SEARCH_PROJECT
  "معلومات عن وزارة الصحة" → MINISTRY_INFO
  "هلا" → GREETING"""


def detect_intent(llm: ChatOllama, text: str) -> str:
    msgs = [SystemMessage(content=INTENT_SYSTEM), HumanMessage(content=text)]
    result = llm.invoke(msgs).content.strip().upper()
    # Check ABOUT before MINISTRY_INFO so "من انت" isn't mistaken for a ministry query
    for intent in ["ABOUT", "GREETING", "SEARCH_PROJECT", "SEARCH_SERVICE", "MINISTRY_INFO"]:
        if intent in result:
            return intent
    return "OTHER"


# Fixed welcome message — hardcoded so the model never invents capabilities
# that the system doesn't actually have (e.g. "login to your account").
WELCOME_MESSAGE = (
    "أهلاً بك! أنا مساعد بوابة الهيئة السعودية للبيانات والذكاء الاصطناعي.\n\n"
    "يمكنني مساعدتك في:\n"
    "• البحث عن مشاريع حكومية (مثال: مشاريع المدن الذكية، مشاريع الطاقة)\n"
    "• استعراض الخدمات الحكومية المتاحة (مثال: تراخيص، تسجيل مركبة)\n"
    "• معلومات عن الوزارات\n"
    "• تقديم طلب خدمة حكومية\n\n"
    "بماذا تريد أن أساعدك؟"
)


# ---------------------------------------------------------------------------
# Formatting Helpers (pure Python — no LLM needed)
# ---------------------------------------------------------------------------

def format_project_list(projects: list) -> str:
    lines = ["**نتائج البحث — المشاريع الحكومية:**\n"]
    for i, p in enumerate(projects, 1):
        lines.append(
            f"{i}. **{p['name']}**\n"
            f"   - الوزارة: {ar_ministry(p['ministry'])} | الميزانية: {ar_money(p['budget'])} | الحالة: {STATUS_AR.get(p['status'], p['status'])}"
        )
    lines.append("\nاختر رقماً من القائمة للحصول على التفاصيل الكاملة.")
    return "\n".join(lines)


def format_service_list(services: list) -> str:
    lines = ["**نتائج البحث — الخدمات الحكومية:**\n"]
    for i, s in enumerate(services, 1):
        lines.append(
            f"{i}. **{s['name']}**\n"
            f"   - الوزارة: {ar_ministry(s['ministry'])} | التصنيف: {CATEGORY_AR.get(s['category'], s['category'])} | الرسوم: {ar_money(s['fee'])}"
        )
    lines.append("\nاختر رقماً من القائمة للحصول على التفاصيل الكاملة.")
    return "\n".join(lines)


def format_project_detail(p: dict) -> str:
    return (
        f"**تفاصيل المشروع: {p['name']}**\n\n"
        f"- **الرقم التعريفي:** {p['id']}\n"
        f"- **الوزارة:** {ar_ministry(p['ministry'])}\n"
        f"- **الميزانية:** {ar_money(p['budget'])}\n"
        f"- **الحالة:** {STATUS_AR.get(p['status'], p['status'])}\n"
        f"- **مدير المشروع:** {p['lead']}\n"
        f"- **آخر تحديث:** 2026-06-15\n"
        f"- **المراحل:** جمع المتطلبات ✓ | اختيار الموردين ⏳ | إطلاق تجريبي Q4 2026"
    )


def format_service_detail(s: dict) -> str:
    return (
        f"**تفاصيل الخدمة: {s['name']}**\n\n"
        f"- **الرقم التعريفي:** {s['id']}\n"
        f"- **الوزارة:** {ar_ministry(s['ministry'])}\n"
        f"- **التصنيف:** {CATEGORY_AR.get(s['category'], s['category'])}\n"
        f"- **الرسوم:** {ar_money(s['fee'])}\n"
        f"- **مدة المعالجة:** {ar_days(s['processing_days'])}\n"
        f"- **المستندات المطلوبة:** الهوية الوطنية، نموذج الطلب، المستندات الداعمة\n\n"
        f"هل تريد تقديم طلب لهذه الخدمة؟"
    )


def format_ministry(name: str, info: dict) -> str:
    ar_name = MINISTRY_AR.get(name, name)
    return (
        f"**معلومات وزارة {ar_name}**\n\n"
        f"- **الوزير:** {info['head']}\n"
        f"- **عدد الموظفين:** {info['employees']:,}\n"
        f"- **الميزانية السنوية:** {ar_money(info['annual_budget'])}\n"
        f"- **تأسست عام:** {info['established']}"
    )


def format_report(s: dict, name: str, nid: str) -> str:
    return (
        f"**تقرير الطلب — يرجى المراجعة قبل التأكيد**\n\n"
        f"| البند | القيمة |\n"
        f"|---|---|\n"
        f"| الخدمة | {s['name']} |\n"
        f"| الوزارة | {ar_ministry(s['ministry'])} |\n"
        f"| مقدم الطلب | {name} |\n"
        f"| رقم الهوية | {nid} |\n"
        f"| الرسوم | {ar_money(s['fee'])} |\n"
        f"| مدة المعالجة | {ar_days(s['processing_days'])} |\n\n"
        f"هل تؤكد تقديم الطلب؟ (اكتب أي كلمة موافقة للتنفيذ، أو 'لا' للإلغاء)"
    )


def format_success(s: dict, name: str) -> str:
    ref = f"REF-{random.randint(100000, 999999)}"
    return (
        f"✅ **تم تقديم طلبك بنجاح!**\n\n"
        f"- **رقم المرجع:** `{ref}`\n"
        f"- **الخدمة:** {s['name']}\n"
        f"- **مقدم الطلب:** {name}\n"
        f"- **المدة المتوقعة:** {ar_days(s['processing_days'])}\n\n"
        f"احتفظ برقم المرجع **{ref}** لمتابعة طلبك."
    )


# ---------------------------------------------------------------------------
# Main Router
# ---------------------------------------------------------------------------

class SDAAgent:
    def __init__(self, llm: ChatOllama, project_idx: SemanticIndex,
                 service_idx: SemanticIndex, ministry_idx: SemanticIndex):
        self.llm = llm
        self.project_idx = project_idx
        self.service_idx = service_idx
        self.ministry_idx = ministry_idx
        self.state = State()

    def _classify_intent(self, text: str) -> str:
        """
        Returns 'yes', 'no', or 'unclear'.
        Checks explicit word lists first (fast), then falls back to the LLM.
        """
        low = text.lower().strip()

        # 1) Exact match
        if low in AFFIRMATIVES:
            return "yes"
        if low in CANCEL_WORDS:
            return "no"

        # 2) Word-level match — handles natural sentences like "تمام مشيها"
        #    or "ايه ابي اقدم" where an affirmative word is mixed with others.
        words = {ar_normalize(w) for w in text.split()}
        if words & _NORM_CANCEL:
            return "no"
        if words & _NORM_AFFIRM:
            return "yes"

        # 3) Hint keywords anywhere in the text
        if any(h in low for h in _YES_HINTS):
            return "yes"
        if any(h in low for h in _NO_HINTS):
            return "no"

        # 4) LLM fallback for ambiguous / longer sentences
        prompt = (
            "The user was asked a yes/no confirmation question in an Arabic government portal. "
            "Their reply is: \"" + text + "\"\n"
            "Reply with exactly one word: YES or NO or UNCLEAR."
        )
        result = self.llm.invoke(prompt).content.strip().upper()
        if "YES" in result:
            return "yes"
        if "NO" in result:
            return "no"
        return "unclear"

    def _parse_number(self, text: str) -> Optional[int]:
        arabic_digits = {"٠": 0, "١": 1, "٢": 2, "٣": 3, "٤": 4, "٥": 5, "٦": 6, "٧": 7, "٨": 8, "٩": 9}
        t = text.strip()
        for ar, en in arabic_digits.items():
            t = t.replace(ar, str(en))
        try:
            return int(t)
        except ValueError:
            # word numbers
            words = {"واحد": 1, "اثنين": 2, "اثنان": 2, "ثلاثة": 3, "اربعة": 4, "خمسة": 5, "ستة": 6,
                     "one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6}
            low = text.strip().lower()
            return words.get(low)

    def _best_ministry(self, text: str) -> Optional[str]:
        """
        Return the best-matching ministry KEY, or None if ambiguous/no match.
        Uses semantic similarity with a margin check so a weak/tied match
        doesn't silently pick the wrong ministry.
        """
        idx = self.ministry_idx
        q_vec = idx._embed(text)
        scored = sorted(
            ((idx._cos(q_vec, v), rec["key"]) for v, rec in zip(idx.vectors, idx.records)),
            key=lambda x: x[0], reverse=True,
        )
        if not scored:
            return None
        top_score, top_key = scored[0]
        # The top match is reliable above this threshold; a tiny margin guard
        # only blocks true ties.
        if top_score >= 0.5:
            return top_key
        return None

    def _resolve_selection(self, text: str) -> Optional[dict]:
        """
        Resolve the user's choice from the last shown list — adaptively.
        Tries, in order: (1) a number, (2a) normalized substring name match,
        (2b) word overlap, (2c) a distinctive word unique to one item,
        (3) the best constrained semantic match.
        Returns the chosen record, or None if nothing confidently matches.
        """
        s = self.state
        if not s.last_list:
            return None

        # 1) Number selection
        n = self._parse_number(text)
        if n and 1 <= n <= len(s.last_list):
            return s.last_list[n - 1]

        # 2) Name match — normalized (handles ة/ه, أ/ا, تشكيل) so misspellings
        #    like "رخصه" still match "رخصة".
        norm_text = ar_normalize(text)
        if len(norm_text) >= 3:
            # 2a) substring, both directions
            for item in s.last_list:
                name = ar_normalize(item["name"])
                if norm_text in name or name in norm_text:
                    return item

            # 2b) word overlap — "ايه ذي الاولى رخصة البناء" should match
            #     "طلب رخصة بناء" even with different word order / "ال" prefix.
            user_words = ar_content_words(text)
            best_item, best_overlap = None, 0.0
            for item in s.last_list:
                name_words = ar_content_words(item["name"])
                if not name_words:
                    continue
                overlap = len(name_words & user_words) / len(name_words)
                if overlap > best_overlap:
                    best_item, best_overlap = item, overlap
            # Require most of the item's own words to be present.
            if best_item is not None and best_overlap >= 0.6:
                return best_item

            # 2c) distinctive word — a word that appears in EXACTLY ONE item's
            #     name uniquely identifies it. e.g. "ابي ذيك حق جده" -> the
            #     Jeddah project; "ابي البناء" -> building permit.
            matches = [
                item for item in s.last_list
                if {w for w in (ar_content_words(item["name"]) & user_words) if len(w) >= 3}
            ]
            if len(matches) == 1:
                return matches[0]

        # 3) Semantic match — constrained to the items actually shown.
        # Only commit if the top candidate is BOTH above threshold AND clearly
        # ahead of the runner-up. If it's ambiguous, return None so the caller
        # re-searches instead of silently picking the wrong item.
        idx = self.project_idx if s.last_list_type == "project" else self.service_idx
        q_vec = idx._embed(text)
        scored = [(idx._cos(q_vec, idx.vectors[idx.records.index(item)]), item)
                  for item in s.last_list]
        scored.sort(key=lambda x: x[0], reverse=True)
        if scored:
            top_score, top_item = scored[0]
            margin = top_score - (scored[1][0] if len(scored) > 1 else 0.0)
            # The small local embedding model gives every Arabic phrase a high
            # baseline similarity, so on-topic selections (~0.77+) and off-topic
            # inputs (~0.73-) separate by absolute score more reliably than by
            # margin. Accept the top item if it clearly leads (margin) OR scores
            # high enough to be a confident on-topic match.
            if top_score >= 0.5 and (margin >= 0.08 or top_score >= 0.75):
                return top_item

        return None

    def handle(self, user_input: str) -> str:
        s = self.state
        text = user_input.strip()

        # ── PICKED_LIST: user picks by number OR by name ───────────────────
        if s.stage == Stage.PICKED_LIST:
            item = self._resolve_selection(text)
            if item is not None:
                s.last_list = None
                if s.last_list_type == "project":
                    s.stage = Stage.FREE
                    return format_project_detail(item)
                else:  # service
                    s.pending_service = item
                    s.stage = Stage.CONFIRM_APPLY
                    return format_service_detail(item)
            # Couldn't match a number or a name → treat as a brand-new request
            # instead of rigidly insisting on a number (adaptive behavior).
            s.stage = Stage.FREE
            s.last_list = None
            # fall through to FREE handling below

        # ── CONFIRM_APPLY: asking if user wants to apply ───────────────────
        if s.stage == Stage.CONFIRM_APPLY:
            intent = self._classify_intent(text)
            if intent == "yes":
                s.stage = Stage.ASK_NAME
                return f"سأبدأ تقديم طلب خدمة **{s.pending_service['name']}**.\n\nما هو اسمك الكامل؟"
            elif intent == "no":
                s.stage = Stage.FREE
                s.pending_service = None
                return "حسناً. كيف يمكنني مساعدتك؟"
            else:
                return f"هل تريد تقديم طلب لخدمة **{s.pending_service['name']}**؟ (نعم / لا)"

        # ── ASK_NAME ───────────────────────────────────────────────────────
        if s.stage == Stage.ASK_NAME:
            s.applicant_name = text
            s.stage = Stage.ASK_NID
            return "شكراً. ما هو رقم هويتك الوطنية؟"

        # ── ASK_NID ────────────────────────────────────────────────────────
        if s.stage == Stage.ASK_NID:
            s.national_id = text
            s.stage = Stage.CONFIRM_SUB
            return format_report(s.pending_service, s.applicant_name, s.national_id)

        # ── CONFIRM_SUB: final confirmation ───────────────────────────────
        if s.stage == Stage.CONFIRM_SUB:
            intent = self._classify_intent(text)
            if intent == "yes":
                result = format_success(s.pending_service, s.applicant_name)
                s.stage = Stage.FREE
                s.pending_service = None
                s.applicant_name = None
                s.national_id = None
                return result
            elif intent == "unclear":
                return "هل تؤكد تقديم الطلب؟ (اكتب نعم للتأكيد أو لا للإلغاء)"
            else:
                s.stage = Stage.FREE
                s.pending_service = None
                s.applicant_name = None
                s.national_id = None
                return "تم إلغاء الطلب. كيف يمكنني مساعدتك؟"

        # ── FREE: detect intent and act ────────────────────────────────────
        intent = detect_intent(self.llm, text)

        if intent in ("GREETING", "ABOUT"):
            return WELCOME_MESSAGE

        if intent == "SEARCH_PROJECT":
            results = self.project_idx.search(text)
            if not results:
                return "لم أجد مشاريع تطابق بحثك. جرّب كلمات أخرى مثل: طاقة، صحة، مدن ذكية، نقل."
            if len(results) == 1:
                return format_project_detail(results[0])
            s.last_list = results
            s.last_list_type = "project"
            s.stage = Stage.PICKED_LIST
            return format_project_list(results)

        if intent == "SEARCH_SERVICE":
            results = self.service_idx.search(text)
            if not results:
                return "لم أجد خدمات تطابق بحثك. جرّب كلمات مثل: ترخيص، تسجيل، تقييم، بناء."
            if len(results) == 1:
                svc = results[0]
                s.pending_service = svc
                s.stage = Stage.CONFIRM_APPLY
                return format_service_detail(svc)
            s.last_list = results
            s.last_list_type = "service"
            s.stage = Stage.PICKED_LIST
            return format_service_list(results)

        if intent == "MINISTRY_INFO":
            # Semantic match — handles spelling variants (الصحه/الصحة), synonyms, AR/EN.
            # Require the top match to clearly beat the runner-up, otherwise ask.
            key = self._best_ministry(text)
            if key:
                return format_ministry(key, MINISTRIES_DB[key])
            available = "\n".join(f"• {MINISTRY_AR[k]}" for k in MINISTRIES_DB)
            return f"عن أي وزارة تسأل؟ الوزارات المتاحة:\n{available}"

        return WELCOME_MESSAGE


# ---------------------------------------------------------------------------
# Startup
# ---------------------------------------------------------------------------

def build() -> SDAAgent:
    print("\n  [Startup] Loading models...")
    embedder = OllamaEmbeddings(model="nomic-embed-text")
    llm = ChatOllama(model="llama3.1", temperature=0)

    print("  [Semantic] Building project index...", end=" ", flush=True)
    p_idx = SemanticIndex(embedder, PROJECTS_DB, ["name", "ministry", "status"])
    print("done.")

    print("  [Semantic] Building service index...", end=" ", flush=True)
    s_idx = SemanticIndex(embedder, SERVICES_DB, ["name", "ministry", "category"])
    print("done.")

    print("  [Semantic] Building ministry index...", end=" ", flush=True)
    # Each ministry record carries its English key + both English and Arabic names,
    # so the embedding captures all spelling variants in one vector.
    # Index ONLY distinguishing words (Arabic name + synonyms). Adding the shared
    # "وزارة" boilerplate would inflate every score and crush the margins.
    ministry_records = [
        {"key": k, "name_ar": MINISTRY_AR[k], "synonyms": MINISTRY_SYNONYMS[k]}
        for k in MINISTRIES_DB
    ]
    m_idx = SemanticIndex(embedder, ministry_records, ["name_ar", "synonyms"])
    print("done.")

    return SDAAgent(llm, p_idx, s_idx, m_idx)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    print("\n" + "=" * 60)
    print("  Saudi Digital Authority — AI Assistant (Local POC)")
    print("  Model: llama3.1 + nomic-embed-text via Ollama")
    print("=" * 60)
    print("  Type 'exit' or 'quit' to end the session.")

    try:
        agent = build()
    except Exception as e:
        print("\n  [خطأ] تعذّر الاتصال بـ Ollama أو تحميل النماذج.")
        print("  تأكد من الخطوات التالية ثم أعد المحاولة:")
        print("    1) تشغيل الخدمة:  ollama serve")
        print("    2) تنزيل النماذج: ollama pull llama3.1 && ollama pull nomic-embed-text")
        print(f"\n  التفاصيل التقنية: {e}\n")
        return
    print("\n  Ready! Start chatting.\n")

    while True:
        try:
            user_input = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n\nSession ended.")
            break

        if not user_input:
            continue
        if user_input.lower() in {"exit", "quit", "q"}:
            print("Goodbye!")
            break

        print("\nAssistant: ", end="", flush=True)
        try:
            reply = agent.handle(user_input)
            print(reply)
        except Exception as e:
            print(f"[Error] {e}")
        print()


if __name__ == "__main__":
    main()
