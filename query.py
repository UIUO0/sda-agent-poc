# -*- coding: utf-8 -*-
"""
query.py — حلقة تفاعلية: توجيه ذكي (دردشة/بحث) → استرجاع → إجابة مقيدة بالسياق.
الاستخدام:
  python3 query.py              # حلقة تفاعلية
  python3 query.py "سؤالك هنا"  # سؤال واحد ثم خروج
"""
import os
import re
import sys
from pathlib import Path

import chromadb
import ollama

BASE_DIR = Path(__file__).parent
DB_DIR = BASE_DIR / "chroma_db"

EMBED_MODEL = os.environ.get("RAG_EMBED_MODEL", "bge-m3")
LLM_MODEL = os.environ.get("RAG_LLM_MODEL", "qwen3.5:9b")
COLLECTION = "documents"
# أقصى مسافة كوساين تُعتبر معها النتيجة ذات صلة (قابلة للضبط)
MAX_DISTANCE = float(os.environ.get("RAG_MAX_DISTANCE", "0.75"))

# مستويات التفكير: السياق سخي دائمًا، والفرق الأساسي في طول الرد وأسلوبه (هندسة برومبت)
DEPTHS = {
    "fast":     {"k": 6, "expand": 0,
                 "style": "اختصر اختصارًا شديدًا: أجب في جملتين إلى ثلاث جمل فقط، بلا مقدمات."},
    "balanced": {"k": 8, "expand": 1,
                 "style": "اختصار متوسط: فقرة موجزة وافية أو نقاط قليلة مركزة."},
    "deep":     {"k": 12, "expand": 2,
                 "style": "فصّل الإجابة: غطِّ كل الجوانب الواردة في المقاطع، ونظّمها بعناوين ونقاط."},
    "max":      {"k": 12, "expand": 2, "verify": True,
                 "style": "فصّل تفصيلًا شاملًا منظمًا دون إسقاط أي عنصر."},
}
DEPTH_NAMES = {"سريع": "fast", "متوازن": "balanced", "عميق": "deep", "دقيق": "max"}
# هامش قبول ملف إضافي: لا يدخل ملف ثانٍ إلا إذا كانت أفضل مقاطعه قريبة من أفضل ملف
FILE_MARGIN = float(os.environ.get("RAG_FILE_MARGIN", "0.10"))
# تعزيز الملف الذي يَرِد اسمه (أو جزء منه) في السؤال — لكل كلمة مطابقة
NAME_BOOST = float(os.environ.get("RAG_NAME_BOOST", "0.06"))
# تعزيز المقطع الذي يطابق عنوان قسمه كلمات السؤال — لكل كلمة مطابقة
HEAD_BOOST = float(os.environ.get("RAG_HEAD_BOOST", "0.08"))

_AR_NORM = str.maketrans({"أ": "ا", "إ": "ا", "آ": "ا", "ة": "ه", "ى": "ي", "ؤ": "و", "ئ": "ي"})
_STOPWORDS = {"وش", "ايش", "هل", "كيف", "ليش", "لماذا", "متى", "اين", "عند", "على",
              "الى", "من", "في", "عن", "بين", "لدى", "خلال", "بعد", "قبل", "حول",
              "the", "and", "for", "with", "what", "how", "when", "letter", "file", "pdf"}


def _keywords(text: str):
    """كلمات مفتاحية منسّقة (توحيد الهمزات والتاء المربوطة، إزالة التطويل والتشكيل)."""
    text = re.sub(r"[ـً-ٰٟ]", "", text.lower())  # كشيدة + تشكيل
    return {w.translate(_AR_NORM) for w in re.findall(r"\w+", text)
            if len(w) >= 3 and w not in _STOPWORDS}

NOT_FOUND_MSG = "لم أجد الإجابة في مستندات الهيئة التي لديك صلاحية الوصول إليها."

SYSTEM_PROMPT = f"""أنت مساعد ذكي يجيب باللغة العربية الفصحى فقط اعتمادًا على مقاطع من مستندات المستخدم.
كل مقطع في «السياق» يحمل درجة مطابقة مئوية تقدّر احتمال صلته بالسؤال.
قواعد يجب الالتزام بها:
1. اعتمد على المقاطع الأعلى درجةً أولًا، واستخرج منها أفضل إجابة ممكنة.
2. كل مقطع يذكر «القسم» الذي ينتمي إليه. أجب حصريًا من المقاطع التي ينتمي قسمها لموضوع السؤال، ولا تخلط أبدًا محتوى قسم أو سياسة أخرى في الإجابة حتى لو ورد في السياق — وإن استشهدت بقسم مختلف فصرّح باسمه صراحة.
3. إذا كانت المقاطع تتناول موضوع السؤال ولو جزئيًا فأجب منها، ووضّح ما لم يرد ذكره في المستندات إن لزم — لا ترفض الإجابة ما دام الموضوع مطروقًا في المقاطع.
   وإذا كان السؤال عن سياسة أو قسم كامل فلا تكتفِ بالتعريف: غطِّ عناصره الرئيسة كما وردت في المقاطع (النطاق، المبادئ، الخطوات والإجراءات، الضوابط) بقدر ما يسمح أسلوب الطول المطلوب.
4. إذا لم تجد الإجابة في المقاطع المرفقة (كانت خارج الموضوع أو لا تحتوي المعلومة)، فقل حرفيًا وفقط: «{NOT_FOUND_MSG}» دون أي إضافة أو اجتهاد من معرفتك العامة.
5. لا تنسب للمستندات معلومة ليست فيها، واذكر في نهاية إجابتك: المصدر: اسم_الملف (والصفحات والقسم).
6. التزم بأسلوب الطول المطلوب في نهاية هذه التعليمات بدقة."""

SECTION_PROMPT = """أنت "مستخرج بيانات صارم" (Strict Data Extractor). مهمتك استخراج وتلخيص المعلومات من السياق المرفق الخاص بقسم «{section}» باللغة العربية الفصحى.
السياق يحتوي على مقاطع مرقمة. اقرأها بالكامل، ثم أخرج إجابتك مهيكلة ومنظمة، مع الالتزام الحرفي بالقواعد التالية:
1) الهيكلة الأصلية: التزم بهيكلة النص الأصلي. إذا كان النص مقسماً إلى (نطاق، مبادئ، خطوات، ضوابط، إلخ)، حافظ على هذا التقسيم ولا تدمج محتوى قسم مع آخر.
2) دقة الكيانات (Entities): انقل أسماء الأشخاص، الجهات، أو الإدارات كما هي نصاً دون أي تعديل أو خلط بين المسميات المتشابهة.
3) الأرقام والمقاييس: استخرج أي أرقام، مدد زمنية، تواريخ، أو إحصائيات بدقة متناهية وبدون تقريب.
4) التجرد التام: يمنع منعاً باتاً استنتاج أو إضافة أي معلومة من خارج السياق المرفق.
اذكر في النهاية: المصدر: اسم_الملف (الصفحات — القسم)."""

VERIFY_PROMPT = """أنت "مدقق جودة صارم" (Strict Quality Auditor). أمامك النص الأصلي من الوثيقة ومسودة إجابة.
أعد كتابة الإجابة النهائية وتصحيح المسودة بناءً على القواعد التالية:
1. التطابق الرقمي: تحقق من نقل أي أرقام أو مدد زمنية أو نسب مئوية بدقة مطلقة. إذا أهملتها المسودة فيجب إضافتها فوراً من النص الأصلي.
2. دقة المسميات: تأكد من عدم الخلط بين الكيانات أو الجهات المذكورة في النص (كأن تخلط بين جهتين أو إدارتين متشابهتين بالاسم). صحح المسميات لتطابق النص الأصلي حرفياً.
3. إزالة الهلوسة: احذف أي استنتاج أو ادعاء أو شرح إضافي غير موجود بشكل صريح في النص الأصلي. لكن لا تحذف أي مدد زمنية أو أرقام وردت في النص الأصلي، وتأكد من تضمينها في الإجابة النهائية — فالحذف يقتصر على الإضافات المستنتجة فقط ولا يشمل الوقائع الرقمية إطلاقاً.
4. «المقتطفات الحرفية» المرفقة في السياق هي مرجعك الأعلى، اعتمدها كما هي.
أخرج الإجابة النهائية المصححة فقط بالعربية الفصحى، واذكر المصدر والصفحات في النهاية."""

def _extract_anchors(hits):
    """مراسٍ حتمية تُستخرج بتعابير نمطية من نص القسم (بلا نموذج = بلا هلوسة):
    أسماء المبادئ (أول ظهور)، النطاق واستثناءاته، والجمل الحاوية للمدد الزمنية."""
    text = re.sub(r"ـ", "", "\n".join(d for d, _, _ in hits))
    clean = lambda s: re.sub(r"\s+", " ", s).strip()
    found = []

    def add(s, limit=170):
        s = clean(s)[:limit]
        if s and not any(s in f or f in s for f in found):
            found.append(s)

    # المبادئ بمسمياتها: أول ظهور فقط لكل مبدأ
    seen = set()
    for m in re.finditer(r"المبدأ\s+(ال\S{3,12})\s*:\s*[^\n.،؛:()]{3,60}", text):
        if m.group(1) not in seen:
            seen.add(m.group(1))
            add(m.group(0), 90)

    # النطاق واستثناءاته — النمط يحفظ «لا» النافية إن وُجدت (حرج: عكسها يقلب المعنى)
    for m in re.finditer(r"(?:لا\s+|كما\s+لا\s+)?تنطبق\s+أحكام[^.؛]{5,160}", text):
        add(m.group(0))

    # المدد الزمنية: فقط بصيغة سليمة الترتيب (رقم بعد «لا تتجاوز/خلال» وقبل كلمة زمنية)
    # — مرساة خاطئة أسوأ من لا مرساة، فالنص المبعثر يُرفض
    for m in re.finditer(
            r"[^.؛\n]{0,70}(?:لا\s+تتجاوز|خلال)\s+[\d٠-٩]{1,3}\s*"
            r"(?:يوم\S{0,3}|أيام|شهر\S{0,3}|أشهر|أسبوع\S{0,3})[^.؛\n]{0,70}", text):
        add(m.group(0))

    return found[:20]

CHAT_PROMPT = """أنت مساعد ودود لنظام محلي للإجابة عن الأسئلة من مستندات المستخدم.
رد باللغة العربية بإيجاز ولطف. إذا حيّاك المستخدم فرد التحية بمثلها دون زيادة
(لا تقل «وعليكم السلام» إلا إذا قال المستخدم «السلام عليكم») واعرض المساعدة.
لا تدّعِ معرفة محتوى المستندات في الدردشة العامة؛ اقترح عليه أن يسألك عنها مباشرة."""

GENERAL_PROMPT = """أنت مساعد ذكي يجيب باللغة العربية الفصحى اعتمادًا على معرفتك العامة فقط، دون الرجوع إلى مستندات المستخدم.
ابدأ ردك بتنبيه موجز بين قوسين: (إجابة من المعرفة العامة — ليست من مستنداتك)، ثم قدّم الإجابة ملتزمًا بأسلوب المستوى المطلوب أدناه بدقة."""

# أسلوب الإجابة في «المعرفة العامة» حسب مستوى التفكير: يتدرّج في الطول والدقة والرسمية
GENERAL_STYLES = {
    "fast": "المستوى: سريع. أجب باختصار شديد في جملتين إلى ثلاث جمل، بأسلوب مباشر وبلغة بسيطة واضحة، دون مقدمات أو تفريعات. اذكر الجوهر فقط.",
    "balanced": "المستوى: متوازن. أجب في فقرة موجزة وافية أو نقاط قليلة مركّزة، بأسلوب واضح شبه رسمي. غطِّ أهم النقاط دون إطالة، وميّز باختصار بين المؤكد والمحتمل.",
    "deep": "المستوى: عميق. فصّل الإجابة وغطِّ الجوانب الرئيسة للموضوع، ونظّمها بعناوين ونقاط عند الحاجة. توخَّ الدقة، وميّز صراحةً بين ما هو مؤكد وما هو محتمل، بأسلوب رسمي منظّم.",
    "max": "المستوى: دقة قصوى. قدّم إجابة شاملة ومنظّمة بأعلى درجة من الرسمية (بأسلوب تقرير رسمي)، مع أقصى عناية بالدقة: وضّح حدود المعلومة، ونبّه صراحةً على أي جزء غير مؤكد أو يحتاج تحققًا، وتجنّب الجزم بما لا تتيقن منه، دون إسقاط أي عنصر جوهري.",
}

ATTACH_PROMPT = """أنت مساعد يجيب باللغة العربية الفصحى اعتمادًا على «المستندات المرفقة» مع الرسالة فقط.
القواعد:
1. أجب من محتوى المستندات المرفقة أدناه حصرًا، ولا تؤلّف أي معلومة من خارجها.
2. انقل الأرقام والمدد الزمنية والتواريخ والأسماء كما وردت حرفيًا دون تقريب أو تغيير.
3. إن لم تجد الإجابة في المرفقات فصرّح بذلك بوضوح.
4. اختم بذكر اسم الملف المرفق الذي استندت إليه."""

FALLBACK_PROMPT = f"""أنت مساعد لنظام إجابة من مستندات المستخدم. جرى البحث في مستنداته ولم يُعثر على مقاطع ذات صلة بسؤاله.
ابدأ ردك بهذه الجملة حرفيًا: «{NOT_FOUND_MSG}»
ثم إن كان بإمكانك تقديم إرشاد أو إجراء عملي مفيد من معرفتك العامة، قدّمه بإيجاز تحت عنوان «اقتراح عام (ليس من مستنداتك):». أجب بالعربية الفصحى."""

ROUTER_PROMPT = """مهمتك تصنيف رسالة مستخدم لنظام يجيب من مستنداته. أجب بكلمة واحدة فقط دون شرح:
- «دردشة»: فقط إذا كانت تحية محضة أو شكرًا أو سؤالًا عن النظام نفسه وقدراته (أمثلة: هلا، شكرًا، وش تقدر تسوي؟، من أنت؟، كيف أستخدمك؟).
- «بحث»: لأي شيء آخر — أي سؤال أو طلب معلومة أو إجراء أو نصيحة، حتى لو بدا عامًا. عند أدنى شك اختر «بحث».

الرسالة: {msg}

التصنيف:"""

# تحيات واضحة تُوجَّه للدردشة فورًا دون استدعاء النموذج
GREETING_RE = re.compile(
    r"^(هلا+( والله)?|مرحبا|مرحبًا|اهلا( وسهلا)?|أهلا( وسهلا)?|أهلًا( وسهلًا)?"
    r"|السلام عليكم( ورحمة الله( وبركاته)?)?|صباح الخير|مساء الخير|هاي|هلو"
    r"|شكرا( لك| جزيلا)?|شكرًا|تسلم|مشكور|يعطيك العافي[هة]"
    r"|hi|hello|hey|thanks|thank you)$", re.IGNORECASE)


def _chat(messages, stream=True, max_tokens=None):
    options = {"temperature": 0.2}
    if max_tokens:
        options["num_predict"] = max_tokens
    kwargs = dict(model=LLM_MODEL, messages=messages, options=options,
                  keep_alive="15m", stream=stream)
    try:
        return ollama.chat(think=False, **kwargs)  # إيقاف وضع التفكير لتسريع الرد
    except TypeError:
        return ollama.chat(**kwargs)


def strip_thinking(text: str) -> str:
    return re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()


def iter_tokens(stream):
    """بث الرموز مع إخفاء وسوم التفكير إن ظهرت في المحتوى."""
    buf, started = "", False
    for part in stream:
        token = part["message"]["content"]
        if started:
            yield token
            continue
        buf += token
        s = buf.lstrip()
        if s.startswith("<think>"):
            if "</think>" in s:
                yield s.split("</think>", 1)[1].lstrip("\n")
                started = True
        elif not "<think>".startswith(s[:7]):
            yield buf
            started = True


def greeting_reply(question: str) -> str:
    """رد فوري مطابق لنوع التحية — بدون استدعاء النموذج."""
    q = question
    tail = " كيف أقدر أساعدك؟ اسألني عن أي شيء في مستنداتك."
    if "السلام عليكم" in q:
        return "وعليكم السلام ورحمة الله وبركاته!" + tail
    if "صباح" in q:
        return "صباح النور!" + tail
    if "مساء" in q:
        return "مساء النور!" + tail
    if any(w in q for w in ("شكر", "تسلم", "مشكور", "العافي", "thank")):
        return "العفو! في الخدمة دائمًا — اسألني متى ما احتجت."
    return "أهلًا وسهلًا!" + tail


def route(question: str) -> str:
    """يقرر: 'greet' لتحية واضحة (رد فوري)، 'chat' لدردشة عامة، 'rag' للبحث (الافتراضي عند الشك)."""
    normalized = re.sub(r"[\s!؟?.،,]+", " ", question).strip()
    if GREETING_RE.fullmatch(normalized):
        return "greet"
    resp = _chat([{"role": "user", "content": ROUTER_PROMPT.format(msg=question)}],
                 stream=False, max_tokens=10)
    verdict = strip_thinking(resp["message"]["content"])
    return "chat" if "دردشة" in verdict else "rag"


CONDENSE_PROMPT = (
    "أعد صياغة «السؤال الأخير» ليكون سؤالاً مستقلاً مكتفياً بذاته يُفهم دون الرجوع "
    "إلى المحادثة السابقة، مع استبدال الضمائر والإشارات (مثل: ذلك، هذا، هي) بما تعود إليه. "
    "أعد السؤال المعاد صياغته فقط دون أي شرح أو مقدمة أو علامات اقتباس. "
    "إن كان السؤال مستقلاً أصلاً فأعده كما هو تماماً."
)


def condense_question(history, question: str) -> str:
    """يحوّل سؤال المتابعة إلى سؤال مستقل بالاعتماد على سجل المحادثة — لتحسين الاسترجاع.
    history: قائمة رسائل [{'role': 'user'|'assistant', 'content': ...}] (بلا الرسالة الحالية)."""
    if not history:
        return question
    convo = "\n".join(
        f"{'المستخدم' if m.get('role') == 'user' else 'المساعد'}: {str(m.get('content',''))[:800]}"
        for m in history[-2:])
    try:
        resp = _chat([{"role": "system", "content": CONDENSE_PROMPT},
                      {"role": "user", "content": f"المحادثة:\n{convo}\n\nالسؤال الأخير: {question}\n\nالسؤال المستقل:"}],
                     stream=False, max_tokens=120)
        out = strip_thinking(resp["message"]["content"]).strip().strip('"«»')
        return out or question
    except Exception:
        return question


def _query(col, q_vec, k, where=None):
    kwargs = {"query_embeddings": [q_vec], "n_results": k}
    if where:
        kwargs["where"] = where
    res = col.query(**kwargs)
    return list(zip(res["documents"][0], res["metadatas"][0], res["distances"][0]))


def _name_boosts(col, question: str):
    """تعزيز للملفات التي يتقاطع اسمها مع كلمات السؤال (مثل ذكر اسم جهة في اسم الملف)."""
    q_words = _keywords(question)
    if not q_words:
        return {}
    try:
        metas = col.get(include=["metadatas"])["metadatas"]
    except Exception:
        return {}
    boosts = {}
    for src in {m["source"] for m in metas}:
        overlap = len(q_words & _keywords(src))
        if overlap:
            boosts[src] = min(overlap * NAME_BOOST, 0.25)
    return boosts


def select_hits(col, question: str, cfg, allowed=None):
    """
    اختيار ذكي: يجمع مرشحين من كل الملفات، يرتّب الملفات حسب أفضل مقطع فيها،
    ثم يعتمد أقل عدد ملفات يكفي — لا يُضاف ملف إلا إذا كانت مقاطعه قريبة من الأفضل.
    يُرجع (المقاطع النهائية، عدد الملفات المرشّحة).

    allowed: مجموعة أسماء الملفات المسموح للمستخدم بالوصول إليها (صلاحيات الوصول).
             None = بلا قيد (كل الملفات). مجموعة فارغة = لا وصول لأي ملف.
    """
    k = cfg["k"]   # عدد المقاطع المسترجعة = المرتبط بمستوى التفكير (يضبطه المشرف لكل مستوى)
    where_allow = None if allowed is None else {"source": {"$in": list(allowed)}}
    if allowed is not None and not allowed:
        return [], 0, ""  # لا صلاحية على أي ملف
    q_vec = ollama.embed(model=EMBED_MODEL, input=[question])["embeddings"][0]
    pool = _query(col, q_vec, min(k * 3, 24), where=where_allow)

    # الملفات التي يظهر اسمها في السؤال: نجلب مقاطعها مباشرة ونعزّزها في الترتيب
    boosts = _name_boosts(col, question)
    if allowed is not None:
        boosts = {s: b for s, b in boosts.items() if s in allowed}  # لا تعزيز لملف خارج الصلاحية
    for src in boosts:
        pool.extend(_query(col, q_vec, k, where={"source": src}))

    # إزالة التكرار ثم تطبيق تعزيزَي اسم الملف وعنوان القسم على المسافات
    q_words = _keywords(question)

    def head_boost(meta):
        overlap = len(q_words & _keywords(meta.get("heading", "")))
        return min(overlap * HEAD_BOOST, 0.25)

    seen, uniq = set(), []
    for doc, meta, dist in pool:
        key = (meta["source"], meta.get("chunk"))
        if key not in seen:
            seen.add(key)
            uniq.append((doc, meta,
                         dist - boosts.get(meta["source"], 0.0) - head_boost(meta)))

    relevant = [h for h in uniq if h[2] <= MAX_DISTANCE]
    if not relevant:
        return [], 0, ""

    # تركيز القسم: إذا وُجدت مقاطع يطابق عنوان قسمها السؤال بوضوح، نقتصر عليها
    # لمنع خلط سياسات/أقسام مختلفة في سياق واحد
    section_hits = [h for h in relevant if head_boost(h[1]) > 0]
    if len(section_hits) >= 3:
        relevant = section_hits

    # جلب القسم كاملًا: تطابق قوي مع عنوان قسم = السؤال عن القسم/السياسة نفسها،
    # فنجلب كل مقاطع القسم بترتيبها الأصلي (لا أشباهها فقط) لضمان تغطية شاملة
    if section_hits:
        top = min(section_hits, key=lambda h: h[2])
        if head_boost(top[1]) >= 2 * HEAD_BOOST:
            heading = top[1].get("heading", "")
            sec = _fetch_section(col, top[1]["source"], heading,
                                 cap=max(12, k * 2), dist=top[2])
            if len(sec) >= 2:
                n_files = len({m["source"] for _, m, _d in relevant})
                return sec, max(n_files, 1), heading

    by_file = {}
    for h in relevant:
        by_file.setdefault(h[1]["source"], []).append(h)
    # ترتيب الملفات حسب أقرب مقطع فيها
    ranked = sorted(by_file.items(), key=lambda kv: min(x[2] for x in kv[1]))
    best_score = min(x[2] for x in ranked[0][1])

    hits = []
    for src, file_hits in ranked:
        if hits:
            if len(hits) >= k:                                   # اكتفينا
                break
            if min(x[2] for x in file_hits) > best_score + FILE_MARGIN:
                break                                            # الملف التالي أبعد من الهامش
        hits.extend(file_hits)

    hits = sorted(hits, key=lambda h: h[2])[:k]
    hits = _expand_neighbors(col, hits, cfg.get("expand", 0))
    return hits, len(by_file), ""


def _fetch_section(col, source: str, heading: str, cap: int, dist: float):
    """جلب كل مقاطع قسم معيّن من ملف معيّن بترتيب القراءة (حتى سقف cap)."""
    if not heading:
        return []
    try:
        got = col.get(where={"$and": [{"source": {"$eq": source}},
                                      {"heading": {"$eq": heading}}]})
        sec = sorted(zip(got["documents"], got["metadatas"]),
                     key=lambda x: x[1].get("chunk", 0))[:cap]
        return [(d, m, dist) for d, m in sec]
    except Exception:
        return []


def _expand_neighbors(col, hits, n: int):
    """توسيع سياق أفضل n مقاطع بضم أطراف المقطعين المجاورين (دقة مطابقة + اكتمال سياق)."""
    out = []
    for idx, (doc, meta, dist) in enumerate(hits):
        if idx < n and meta.get("chunk") is not None:
            src, c = meta["source"], meta["chunk"]
            try:
                got = col.get(ids=[f"{src}::{c - 1}", f"{src}::{c + 1}"])
                # الجار يُضم فقط إذا كان من القسم نفسه — لا تسرب بين الأقسام
                near = {i: d for i, d, m in zip(got.get("ids", []),
                                                got.get("documents", []),
                                                got.get("metadatas", []))
                        if (m or {}).get("heading", "") == meta.get("heading", "")}
                prev = near.get(f"{src}::{c - 1}", "")[-500:]
                nxt = near.get(f"{src}::{c + 1}", "")[:500]
                doc = " … ".join(x for x in (prev, doc, nxt) if x)
            except Exception:
                pass
        out.append((doc, meta, dist))
    return out


def match_score(dist: float) -> int:
    """تحويل مسافة كوساين إلى درجة مطابقة مئوية تقريبية."""
    return max(1, min(99, round((1 - dist) * 100)))


def build_context(hits, section: str = ""):
    blocks = []
    for i, (doc, meta, dist) in enumerate(hits, 1):
        src = meta["source"]
        if meta.get("pages"):
            src += f" (ص. {meta['pages']})"
        if section:  # وضع القسم الكامل: ترقيم تسلسلي يوضح أن السياق متصل وكامل
            header = f"[الجزء {i} من {len(hits)} — {src}]"
        else:
            head = f" — القسم: {meta['heading']}" if meta.get("heading") else ""
            header = f"[مقطع {i} — المصدر: {src}{head} — درجة المطابقة: {match_score(dist)}%]"
        blocks.append(f"{header}\n{doc}")
    return "\n\n---\n\n".join(blocks)


def _history_msgs(history, limit=2):
    """سياق المحادثة = التبادل الأخير فقط (آخر سؤال وردّه) — منقّى ومقصوص."""
    out = []
    for m in (history or [])[-limit:]:
        role = "assistant" if m.get("role") == "assistant" else "user"
        content = str(m.get("content", "")).strip()[:1200]
        if content:
            out.append({"role": role, "content": content})
    return out


def build_answer_messages(question: str, hits, section: str, cfg, progress=None, history=None):
    """تجهيز رسائل التوليد النهائي (مشترك بين الطرفية والواجهة).
    وضع الدقة القصوى: مسودة ثم تدقيقها ضد النص الأصلي نفسه.
    history: سجل المحادثة السابق لدعم أسئلة المتابعة (اختياري)."""
    base = SECTION_PROMPT.format(section=section) if section else SYSTEM_PROMPT
    system = base + "\nأسلوب الإجابة المطلوب: " + cfg["style"]

    ctx = build_context(hits, section)
    anchors = _extract_anchors(hits) if section else []
    if anchors:
        ctx += ("\n\nمقتطفات حرفية مستخرجة آليًا من القسم — "
                "انسخ مسمياتها وأرقامها كما هي دون أي تغيير:\n"
                + "\n".join(f"• {a}" for a in anchors))
    user = f"السياق:\n{ctx}\n\nالسؤال: {question}"
    messages = [{"role": "system", "content": system}]
    messages += _history_msgs(history)   # سياق المحادثة السابق (إن وُجد)
    messages.append({"role": "user", "content": user})
    if not cfg.get("verify"):
        return messages

    # المرحلة 1: مسودة كاملة
    if progress:
        progress(1, 2)
    draft = strip_thinking(_chat(messages, stream=False)["message"]["content"])
    # المرحلة 2: تدقيق المسودة ضد النص الأصلي (تُبث للمستخدم)
    if progress:
        progress(2, 2)
    verify_user = (f"النص الأصلي:\n{ctx}\n\n"
                   f"المسودة المطلوب تدقيقها:\n{draft}\n\n"
                   f"السؤال الأصلي: {question}")
    return [{"role": "system", "content": VERIFY_PROMPT},
            {"role": "user", "content": verify_user}]


def _print_stream(stream):
    print()
    for token in iter_tokens(stream):
        print(token, end="", flush=True)
    print()


def answer(col, question: str, depth: str = None):
    depth = depth or os.environ.get("RAG_DEPTH", "balanced")
    cfg = DEPTHS.get(depth, DEPTHS["balanced"])

    mode = route(question)
    if mode == "greet":
        print(f"\n{greeting_reply(question)}")
        return
    if mode == "chat":
        _print_stream(_chat([{"role": "system", "content": CHAT_PROMPT},
                             {"role": "user", "content": question}]))
        return

    hits, n_candidates, section = select_hits(col, question, cfg)

    # لا مقاطع ذات صلة → نصرّح أنها ليست في المستندات ثم نقترح من المعرفة العامة
    if not hits:
        _print_stream(_chat([{"role": "system", "content": FALLBACK_PROMPT},
                             {"role": "user", "content": question}]))
        return

    if section:
        print(f"\n📚 جلب قسم كامل: {section}")
    msgs = build_answer_messages(
        question, hits, section, cfg,
        progress=lambda i, n: print(f"  🎯 استخراج الحقائق {i}/{n}", end="\r"))
    _print_stream(_chat(msgs))
    used_files = sorted({m["source"] for _, m, _d in hits})
    sources = [f"{m['source']}" + (f" (ص. {m['pages']})" if m.get("pages") else "")
               + f" · {match_score(d)}%" for _, m, d in hits[:6]]
    print(f"\n📄 المصادر (بدرجة المطابقة): {'، '.join(sources)}")
    print(f"📁 استخدم {len(used_files)} ملف من أصل {n_candidates} ملف مرشّح — {len(hits)} مقطع")


def main():
    client = chromadb.PersistentClient(path=str(DB_DIR))
    try:
        col = client.get_collection(COLLECTION)
    except Exception:
        sys.exit("لا توجد قاعدة بيانات — شغّل ingest.py أولًا.")

    if len(sys.argv) > 1:
        q = " ".join(sys.argv[1:])
        print(f"\n❓ السؤال: {q}")
        answer(col, q)
        return

    depth = os.environ.get("RAG_DEPTH", "balanced")
    print("اكتب سؤالك (أو «خروج» للإنهاء — ولتغيير مستوى التفكير: عمق سريع / متوازن / عميق / دقيق):")
    while True:
        try:
            q = input("\n❓ سؤالك: ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if not q or q.lower() in ("خروج", "exit", "quit", "q"):
            break
        if q.startswith("عمق"):
            word = q.replace("عمق", "").strip()
            if word in DEPTH_NAMES:
                depth = DEPTH_NAMES[word]
                print(f"✅ مستوى التفكير الآن: {word}")
            else:
                print("الخيارات: عمق سريع / عمق متوازن / عمق عميق")
            continue
        answer(col, q, depth)
    print("\nمع السلامة!")


if __name__ == "__main__":
    main()
