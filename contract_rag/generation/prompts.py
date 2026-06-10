"""
prompts.py — Task-specific prompt templates for ContractRAG generation.
"""

# ------------------------------------------------------------------
# System prompt shared across all tasks
# ------------------------------------------------------------------

SYSTEM_PROMPT = """You are a senior contract lawyer and legal analyst.
You answer questions ONLY using the contract clauses provided in the context below.
Do NOT use any outside knowledge.
If the context does not contain enough information, say so explicitly.
Always respond with valid JSON matching the schema requested.
Never add commentary outside the JSON block."""


# ------------------------------------------------------------------
# Context block builder
# ------------------------------------------------------------------

def build_context_block(chunks: list[dict]) -> str:
    """
    Formats retrieved chunks into a numbered context block
    with citation markers for the LLM to reference.
    """
    lines = []
    for i, chunk in enumerate(chunks, start=1):
        payload = chunk.get("payload", {})
        text = payload.get("text", "").strip()
        source = payload.get("source_file", "unknown")
        page = payload.get("page_number", "?")
        clause = payload.get("clause_type", "general")
        lines.append(
            f"[{i}] (source: {source} | page: {page} | clause: {clause})\n{text}"
        )
    return "\n\n".join(lines)


# ------------------------------------------------------------------
# Task templates
# ------------------------------------------------------------------

ASK_TEMPLATE = """{system}

CONTEXT:
{context}

QUESTION:
{question}

Respond ONLY with a JSON object matching this schema exactly.
The "answer" field MUST be a plain text string summarizing your findings.
{{
  "answer": "<plain text string answer here>",
  "citations": [
    {{"chunk_index": <int>, "clause": "<clause_type>", "page": <int>, "source_file": "<filename>", "score": <float>}}
  ],
  "confidence": <float between 0 and 1>,
  "risk_level": "<HIGH|MEDIUM|LOW|NONE>",
  "flags": ["<any concerns or caveats>"],
  "supporting_chunks": ["<verbatim key phrases from context that support the answer>"]
}}"""


RISK_TEMPLATE = """{system}

CONTEXT (contract clauses):
{context}

TASK:
Analyse the contract clauses above for legal risk. Identify issues in:
- Liability caps (missing, one-sided, or unlimited)
- Indemnification (broad, unilateral, or uncapped)
- Termination triggers (vague, missing notice periods)
- Auto-renewal traps (silent renewal, short opt-out windows)
- Confidentiality gaps (missing obligations, weak definitions)

Respond ONLY with a JSON object matching this schema exactly:
{{
  "answer": "<overall risk summary>",
  "citations": [
    {{"chunk_index": <int>, "clause": "<clause_type>", "page": <int>, "source_file": "<filename>", "score": <float>}}
  ],
  "confidence": <float between 0 and 1>,
  "risk_level": "<HIGH|MEDIUM|LOW|NONE>",
  "flags": ["<specific risk flag per issue found>"],
  "supporting_chunks": ["<verbatim key phrases that evidence each risk>"]
}}"""


COMPARE_TEMPLATE = """{system}

VENDOR CONTRACT CLAUSES:
{vendor_context}

TEMPLATE/BASELINE CONTRACT CLAUSES:
{template_context}

TASK:
Compare the vendor contract against the template clause by clause.
For each clause type present in either contract, identify:
- What the vendor says
- What the template says
- The risk delta (vendor is MORE risky, LESS risky, or EQUIVALENT vs template)

Respond ONLY with a JSON object matching this schema exactly:
{{
  "answer": "<overall comparison summary>",
  "clause_diffs": [
    {{
      "clause_type": "<clause type>",
      "vendor_summary": "<what vendor contract says>",
      "template_summary": "<what template says>",
      "risk_delta": "<MORE_RISKY|LESS_RISKY|EQUIVALENT>",
      "recommendation": "<what to negotiate or flag>"
    }}
  ],
  "citations": [
    {{"chunk_index": <int>, "clause": "<clause_type>", "page": <int>, "source_file": "<filename>", "score": <float>}}
  ],
  "confidence": <float between 0 and 1>,
  "risk_level": "<HIGH|MEDIUM|LOW|NONE>",
  "flags": ["<top negotiation flags>"]
}}"""


HEALTH_TEMPLATE = """{system}

CONTEXT (contract clauses):
{context}

TASK:
Score this contract's health from 0 to 100 using these weighted criteria:
- Completeness (25pts): Are all standard clauses present? (liability, termination, confidentiality, indemnification, governing law)
- Risk exposure (30pts): Are liability caps present and reasonable? Is indemnification balanced?
- Clarity (20pts): Are obligations clearly defined with measurable terms?
- Obligation proximity (25pts): Are key dates, deadlines, and renewal windows clearly stated?

Deduct points for each missing clause, vague term, or risky provision found.

Respond ONLY with a JSON object matching this schema exactly:
{{
  "answer": "<health summary narrative>",
  "health_score": <integer 0-100>,
  "score_breakdown": {{
    "completeness": <int 0-25>,
    "risk_exposure": <int 0-30>,
    "clarity": <int 0-20>,
    "obligation_proximity": <int 0-25>
  }},
  "citations": [
    {{"chunk_index": <int>, "clause": "<clause_type>", "page": <int>, "source_file": "<filename>", "score": <float>}}
  ],
  "confidence": <float between 0 and 1>,
  "risk_level": "<HIGH|MEDIUM|LOW|NONE>",
  "flags": ["<specific issues that reduced the score>"]
}}"""


# ------------------------------------------------------------------
# Template builder functions
# ------------------------------------------------------------------

def build_ask_prompt(question: str, chunks: list[dict]) -> str:
    context = build_context_block(chunks)
    return ASK_TEMPLATE.format(
        system=SYSTEM_PROMPT,
        context=context,
        question=question,
    )


def build_risk_prompt(chunks: list[dict]) -> str:
    context = build_context_block(chunks)
    return RISK_TEMPLATE.format(
        system=SYSTEM_PROMPT,
        context=context,
    )


def build_compare_prompt(vendor_chunks: list[dict], template_chunks: list[dict]) -> str:
    vendor_context = build_context_block(vendor_chunks)
    template_context = build_context_block(template_chunks)
    return COMPARE_TEMPLATE.format(
        system=SYSTEM_PROMPT,
        vendor_context=vendor_context,
        template_context=template_context,
    )


def build_health_prompt(chunks: list[dict]) -> str:
    context = build_context_block(chunks)
    return HEALTH_TEMPLATE.format(
        system=SYSTEM_PROMPT,
        context=context,
    )