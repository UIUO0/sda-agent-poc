# SDA AI Agent — POC

> نموذج أولي لوكيل ذكاء اصطناعي محلي لبوابة الهيئة السعودية للبيانات والذكاء الاصطناعي.

A fully local conversational AI agent for a Saudi government portal. Handles Arabic-language queries for government projects, services, and ministry information — with an end-to-end service application flow.

---

## Features

- **Arabic-first**: understands colloquial Saudi Arabic, spelling variants (ة/ه), and synonyms
- **Semantic search**: finds projects, services, and ministries by meaning — not just keywords
- **Adaptive selection**: users pick items by number, name, or natural description
- **Service application flow**: guided multi-step form (name → national ID → confirmation → success)
- **Fully local**: no cloud API — runs entirely on your machine via Ollama

---

## Architecture

```
User Input
    │
    ▼
Python State Machine  ◄────────────────────────────────────────────┐
    │                                                               │
    ├── FREE stage → LLM intent classifier (llama3.1)              │
    │       │                                                       │
    │       ├── GREETING / ABOUT  → hardcoded welcome message      │
    │       ├── SEARCH_PROJECT   → SemanticIndex → list / detail   │
    │       ├── SEARCH_SERVICE   → SemanticIndex → list / detail   │
    │       └── MINISTRY_INFO    → ministry SemanticIndex → info   │
    │                                                               │
    ├── PICKED_LIST   → adaptive selection (number / name / embed) │
    ├── CONFIRM_APPLY → yes/no intent classifier                   │
    ├── ASK_NAME      → capture applicant name                     │
    ├── ASK_NID       → capture national ID                        │
    └── CONFIRM_SUB   → final confirmation → success / cancel ─────┘
```

The **state machine owns all flow logic**. The LLM handles only two things:

1. Intent classification (in the `FREE` stage)
2. Ambiguous yes/no resolution (confirmation stages)

This keeps the conversation deterministic and predictable — the 7B model cannot hallucinate a step it doesn't control.

### Semantic Search

Three `SemanticIndex` instances using cosine similarity over `nomic-embed-text` embeddings:

| Index | Fields | Threshold |
|---|---|---|
| Projects | `name`, `ministry`, `status` | 0.25 |
| Services | `name`, `ministry`, `category` | 0.25 |
| Ministries | Arabic name + synonyms only | 0.50 |

The ministry index deliberately excludes the shared word "وزارة" from every record. Including it inflates all scores equally, collapses the margin between ministries, and causes wrong matches. Only distinguishing words are indexed.

### Adaptive Selection

When a list is displayed, `_resolve_selection()` matches the user's reply
adaptively — the user never has to reply with just a number. All text is first
passed through `ar_normalize()` (unifies ة/ه, أ/إ/آ→ا, ى/ي, strips diacritics)
so colloquial misspellings still match:

1. **Number** — Arabic or Western digits, or word numbers (واحد, اثنين...)
2. **Normalized substring** — name match in either direction
3. **Word overlap** — picks an item when ≥ 60% of its name words appear in the
   input (handles loose word order and the "ال" prefix, e.g. "البناء" ≈ "بناء")
4. **Distinctive word** — a word appearing in exactly one item's name uniquely
   selects it (e.g. "ابي ذيك حق جده" → the Jeddah project)
5. **Semantic** — cosine similarity against the shown items. Handles synonyms
   and verb forms that share no words with the name (e.g. "بسجل سياره" → "تسجيل
   مركبة"). The top item is accepted when it either clearly leads the runner-up
   (margin ≥ 0.08) or scores high enough on its own (≥ 0.75) — the absolute
   score separates on-topic selections (~0.77+) from off-topic input (~0.73−)
   more reliably than the margin alone, given the small local embedding model.

Confirmation replies (yes/no) are matched the same way: exact word, then
word-level membership in the affirmative/cancel sets (so "تمام مشيها" and
"ايه ابي اقدم" both count as yes), then keyword hints, then an LLM fallback.

---

## Tech Stack

| Component | Tool |
|---|---|
| LLM inference | [Ollama](https://ollama.com) — `llama3.1` (7B) |
| Embeddings | Ollama — `nomic-embed-text` |
| LLM framework | LangChain + LangChain-Ollama |
| Language | Python 3.9+ |

---

## Prerequisites

1. **Python 3.9+**
2. **[Ollama](https://ollama.com/download)** installed and running
3. Required models pulled:

```bash
ollama pull llama3.1
ollama pull nomic-embed-text
```

---

## Installation

```bash
# Clone the repo
git clone https://github.com/your-username/sda-agent-poc.git
cd sda-agent-poc

# Create and activate a virtual environment
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

---

## Running

```bash
# Ensure Ollama is running (starts automatically on macOS after install)
ollama serve

# Launch the agent
python main.py
```

Expected startup output:

```
============================================================
  Saudi Digital Authority — AI Assistant (Local POC)
  Model: llama3.1 + nomic-embed-text via Ollama
============================================================
  [Startup] Loading models...
  [Semantic] Building project index... done.
  [Semantic] Building service index... done.
  [Semantic] Building ministry index... done.

  Ready! Start chatting.
```

Type `exit` or `quit` to end the session.

---

## Example Conversations

> The user inputs below are written in natural Saudi colloquial Arabic, with
> spelling mistakes and loose word order — exactly as a real citizen would type.
> The agent normalizes the text, matches by meaning, and never forces the user
> to reply with a list number.

**Browse projects — and pick one by describing it (not by number):**

```
You: وش مشاريع المدن الذكيه عندكم
Assistant: **نتائج البحث — المشاريع الحكومية:**

1. **مبادرة الشبكة الذكية الوطنية**
   - الوزارة: الطاقة | الميزانية: 500 مليون ريال | الحالة: نشط
2. **البنية التحتية للمدن الذكية — الرياض**
   - الوزارة: الشؤون البلدية | الميزانية: 750 مليون ريال | الحالة: نشط
3. **شبكة ترشيد المياه الذكية**
   - الوزارة: البيئة | الميزانية: 80 مليون ريال | الحالة: نشط
4. **البنية التحتية للمدن الذكية — جدة**
   - الوزارة: الشؤون البلدية | الميزانية: 600 مليون ريال | الحالة: قيد التخطيط
5. **السجلات الصحية الذكية الوطنية**
   - الوزارة: الصحة | الميزانية: 300 مليون ريال | الحالة: قيد التخطيط
6. **نظام إدارة المرور الذكي**
   - الوزارة: النقل | الميزانية: 120 مليون ريال | الحالة: نشط

اختر رقماً من القائمة للحصول على التفاصيل الكاملة.

You: ابي ذيك حق جده
Assistant: **تفاصيل المشروع: البنية التحتية للمدن الذكية — جدة**

- **الرقم التعريفي:** P006
- **الوزارة:** الشؤون البلدية
- **الميزانية:** 600 مليون ريال
- **الحالة:** قيد التخطيط
- **مدير المشروع:** م. ريم الزهراني
- **آخر تحديث:** 2026-06-15
- **المراحل:** جمع المتطلبات ✓ | اختيار الموردين ⏳ | إطلاق تجريبي Q4 2026
```

**Apply for a service — full flow in colloquial Arabic, no numbers:**

```
You: ابي اطلع رخصه بناء لعماره
Assistant: **نتائج البحث — الخدمات الحكومية:**

1. **طلب رخصة بناء**
   - الوزارة: الشؤون البلدية | التصنيف: الإنشاءات | الرسوم: 2000 ريال
2. **تسجيل رخصة تجارية**
   - الوزارة: التجارة | التصنيف: الأعمال | الرسوم: 500 ريال
3. **تجديد رخصة القيادة**
   - الوزارة: النقل | التصنيف: التسجيل | الرسوم: 400 ريال
4. **ترخيص منشأة صحية**
   - الوزارة: الصحة | التصنيف: التراخيص | الرسوم: 5000 ريال
5. **تسجيل مركبة**
   - الوزارة: النقل | التصنيف: التسجيل | الرسوم: 200 ريال
6. **تقييم الأثر البيئي**
   - الوزارة: البيئة | التصنيف: التقييم | الرسوم: 10000 ريال

اختر رقماً من القائمة للحصول على التفاصيل الكاملة.

You: ايه ذي الاولى رخصة البناء
Assistant: **تفاصيل الخدمة: طلب رخصة بناء**

- **الرقم التعريفي:** S002
- **الوزارة:** الشؤون البلدية
- **التصنيف:** الإنشاءات
- **الرسوم:** 2000 ريال
- **مدة المعالجة:** 14 يوم عمل
- **المستندات المطلوبة:** الهوية الوطنية، نموذج الطلب، المستندات الداعمة

هل تريد تقديم طلب لهذه الخدمة؟

You: ايه ابي اقدم
Assistant: سأبدأ تقديم طلب خدمة **طلب رخصة بناء**.

ما هو اسمك الكامل؟

You: عبدالله محمد الغامدي
Assistant: شكراً. ما هو رقم هويتك الوطنية؟

You: 1098765432
Assistant: **تقرير الطلب — يرجى المراجعة قبل التأكيد**

| البند | القيمة |
|---|---|
| الخدمة | طلب رخصة بناء |
| الوزارة | الشؤون البلدية |
| مقدم الطلب | عبدالله محمد الغامدي |
| رقم الهوية | 1098765432 |
| الرسوم | 2000 ريال |
| مدة المعالجة | 14 يوم عمل |

هل تؤكد تقديم الطلب؟ (اكتب أي كلمة موافقة للتنفيذ، أو 'لا' للإلغاء)

You: تمام مشيها
Assistant: ✅ **تم تقديم طلبك بنجاح!**

- **رقم المرجع:** `REF-483934`
- **الخدمة:** طلب رخصة بناء
- **مقدم الطلب:** عبدالله محمد الغامدي
- **المدة المتوقعة:** 14 يوم عمل

احتفظ برقم المرجع **REF-483934** لمتابعة طلبك.
```

**Ministry info — tolerates misspelling (الصحه → الصحة):**

```
You: معلومات عن وزارة الصحه
Assistant: **معلومات وزارة الصحة**

- الوزير: Dr. Fahad Al-Jalajel
- عدد الموظفين: 120,000
- الميزانية السنوية: 180 مليار ريال
- تأسست عام: 1949
```

---

## Project Structure

```
sda-agent-poc/
├── main.py          # All agent logic — data, semantic index, state machine, CLI
├── requirements.txt # Python dependencies
└── .gitignore
```

Everything lives in a single file to keep the POC self-contained and easy to read.

---

## Mock Data

The agent ships with in-memory mock data — no database required:

| Dataset | Records | Description |
|---|---|---|
| `PROJECTS_DB` | 6 | Government projects across 5 ministries |
| `SERVICES_DB` | 6 | Government services with fees and processing times |
| `MINISTRIES_DB` | 6 | Ministry details: minister, headcount, budget, founding year |

---

## Design Decisions

| Decision | Reason |
|---|---|
| State machine instead of pure LLM loop | 7B models hallucinate steps and fields in multi-turn forms; deterministic code does not |
| Hardcoded welcome message | Prevents the model from inventing capabilities the system doesn't have |
| Ministry index uses only distinguishing words | Shared boilerplate ("وزارة") equalizes all scores and kills margins |
| LLM intent classification before any semantic search | Avoids searching the wrong index; keeps search queries clean |
| Layered yes/no resolution (exact word → word-level set membership → hints → LLM) | Fast for the 95% case; LLM only handles genuinely ambiguous replies |

---

## Limitations

- **Mock data only** — no real government databases are connected
- **Single-user session** — state is in-memory; restarting the process resets it
- **Arabic only** — English queries are understood for intent but replies are in Arabic
- **No authentication** — national ID is accepted as plain text input
- **llama3.1 speed** — first response takes a few seconds on CPU; GPU recommended for real-time feel

---

## License

MIT
