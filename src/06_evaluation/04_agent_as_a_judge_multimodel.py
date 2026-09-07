"""
18. Multi-model Agent-as-a-Judge
================================
13.agent-as-a-Judge.py 와 동일한 agent-as-a-judge 로직(3개 평가 기준 +
concept-level notice + Semantic Scholar 논문 검색 1-round reflection)을
3개 모델로 각각 수행한다.

  - GPT    : gpt-5.4-mini           (OpenAI)   — 최신 경량 (GPT-5.4 mini)
  - Claude : claude-opus-4-8        (Anthropic)— 최상위 성능 (Opus 4.8)
  - Gemini : gemini-3.5-flash       (Google)   — 최신 (Gemini 3.5 Flash)

각 기준은 YES/NO 로 판정하고, 모델별 상세 결과를 따로 저장한 뒤
3개 모델 결과를 하나의 종합 표로 합쳐 저장한다.

검증 대상: Table 1 에 예시로 쓰인 3개 텍스트 (prospective node 27 / 41 / 36).

키는 환경변수(OPENAI_API_KEY / ANTHROPIC_API_KEY / GOOGLE_API_KEY)에서 읽는다.
"""
# Source file in the working repository: code/18.agent_judge_multimodel.py
# Release edits: API keys removed (environment variables), paths made repository-relative.
# --- public-release addition: repository root for relative paths ------------
import os as _os
_ROOT = _os.path.abspath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '..', '..'))
# ---------------------------------------------------------------------------
import os, re, json, time
import pandas as pd
import requests

# ──────────────────────────────────────────────────────────────────────────────
# 0. Keys & Models
# ──────────────────────────────────────────────────────────────────────────────
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
ANTHROPIC_API_KEY = os.environ.get("ANTHROPIC_API_KEY")
GOOGLE_API_KEY = os.environ.get("GOOGLE_API_KEY")

MODELS = {
    "GPT":    {"provider": "openai",    "model": "gpt-5.4-mini"},
    "Claude": {"provider": "anthropic", "model": "claude-opus-4-8"},
    "Gemini": {"provider": "google",    "model": "gemini-3.5-flash"},
}

D = _ROOT

# ──────────────────────────────────────────────────────────────────────────────
# 1. 평가 기준 (13.agent-as-a-Judge.py 와 동일)
# ──────────────────────────────────────────────────────────────────────────────
# CRITERIA = {
#     "Originality & Novelty": (
#         "Does the concept propose a genuinely new angle, combination, or framing "
#         "that is not a trivial restatement of existing work? "
#         "Evaluate by comparing the core proposition against the landscape of prior "
#         "research retrieved via search. A concept may combine known elements in a "
#         "novel way — this still counts as original."
#     ),
#     "Scientific & Logical Consistency": (
#         "Are the concept's foundational assumptions compatible with established "
#         "scientific principles? Is the internal reasoning free of logical "
#         "contradictions? NOTE: The absence of a detailed mechanism or mathematical "
#         "formulation is EXPECTED at concept level and must NOT be treated as a flaw. "
#         "Focus only on whether the stated logic violates known laws or contradicts itself."
#     ),
#     "Problem–Concept Alignment": (
#         "Is the problem the concept addresses clearly identified and significant? "
#         "Does the proposed conceptual direction offer a plausible path toward "
#         "addressing that problem? A concept does not need to fully solve the problem; "
#         "it only needs to point in a defensible direction."
#     ),
#     "Research & Development Potential": (
#         "Does the concept open meaningful avenues for future research or engineering "
#         "exploration? Could it inspire follow-up studies, prototyping, or theoretical "
#         "extensions? Evaluate the fertility of the idea, not whether a roadmap exists."
#     ),
#     "Differentiated Value Proposition": (
#         "Does the concept articulate a clear advantage or unique benefit compared to "
#         "existing or alternative approaches? The advantage may be theoretical "
#         "(e.g., a simpler framing, broader applicability) rather than empirically proven."
#     ),
# }

CRITERIA = {
    "Novelty & Differentiation": (
        "Is the core proposition genuinely new relative to the landscape of "
        "prior research retrieved via search — a new angle, combination, or "
        "framing rather than a trivial restatement? A novel combination of "
        "known elements counts as original. Where possible, identify the "
        "closest retrieved prior work and state what distinguishes the concept "
        "from it. Answer YES only if no single retrieved work substantially "
        "discloses the same core proposition."
    ),
    "Scientific & Logical Soundness": (
        "Are the concept's foundational assumptions compatible with established "
        "scientific principles, and is the internal reasoning free of logical "
        "contradictions? NOTE: The absence of a detailed mechanism or "
        "mathematical formulation is EXPECTED at concept level and must NOT be "
        "treated as a flaw. Answer NO only if the stated logic violates known "
        "laws or contradicts itself; otherwise answer YES."
    ),
    "Significance & Value": (
        "Does the concept address a clearly identifiable and non-trivial "
        "problem, and does it point in a defensible direction that offers some "
        "discernible benefit — whether a comparative advantage over existing "
        "approaches or meaningful potential for follow-up research and "
        "development? The concept does not need to solve the problem or provide "
        "a roadmap. Answer YES only if both the problem and the benefit are "
        "identifiable from the concept itself."
    ),
}
CRIT_KEYS = list(CRITERIA.keys())

GLOBAL_PERSONA = "Senior Technology Forecaster & Concept Evaluator"

CONCEPT_LEVEL_NOTICE = """
═══════════════════════════════════════════════════════════════
IMPORTANT CONTEXT — READ BEFORE EVALUATING
═══════════════════════════════════════════════════════════════
The text below is a TECHNOLOGY CONCEPT — an early-stage idea that exists
purely at the conceptual level. This concept was synthesized from patent
data published between 2020 and 2023, representing a prospective technology
direction grounded in that 2020–2023 patent corpus. At this stage, the
following are NORMAL and must NEVER be penalized:

  • No detailed mechanism design or process specification
  • No experimental data, simulation results, or prototypes
  • No mathematical formulation or quantitative modeling
  • No implementation roadmap or engineering specifications
  • Use of high-level or abstract language to describe the approach

Your task is to evaluate the IDEA ITSELF — its logical direction,
theoretical grounding, and potential — not the maturity of its execution.
═══════════════════════════════════════════════════════════════
"""

# Gemini 전용 notice: 위 기본 notice + 문면(문법/반복/순환) 지적 금지 지시.
# Gemini 는 자동 생성 텍스트의 표면 표현을 문면 그대로 읽어 컨셉까지 감점하는 경향이
# 강했으므로(원본 기준 overall 22/55), 해당 지시를 추가한 조건으로 평가한다.
CONCEPT_LEVEL_NOTICE_GEMINI = CONCEPT_LEVEL_NOTICE.replace(
    "  • Use of high-level or abstract language to describe the approach\n",
    "  • Use of high-level or abstract language to describe the approach\n"
    "  • Grammatical errors, awkward/repetitive/circular wording, or truncated\n"
    "    sentences (these are text-generation artifacts, not concept flaws)\n"
).replace(
    "\nYour task is to evaluate the IDEA ITSELF",
    """
NOTE — DO NOT CRITIQUE GRAMMAR OR WORDING:
This text was automatically generated and may contain grammatical errors,
repeated or circular sentences, truncation, or awkward phrasing. Do NOT
raise these surface/language issues as criticisms, and do NOT lower any
verdict because of them. Evaluate the technology concept EXACTLY AS STATED —
do NOT reinterpret or generalize it into a broader or different idea, and do
NOT treat garbled wording as evidence that the concept is flawed. Reserve NO
strictly for substantive problems in the IDEA itself: it contradicts
established science, is self-contradictory at the conceptual level, is a
verbatim duplicate of prior art, or has no identifiable problem or benefit.

Your task is to evaluate the IDEA ITSELF"""
)

# 모델별 notice 매핑 (GPT/Claude = 기본, Gemini = 문면 지적 금지 추가)
NOTICE_BY_MODEL = {"GPT": CONCEPT_LEVEL_NOTICE,
                   "Claude": CONCEPT_LEVEL_NOTICE,
                   "Gemini": CONCEPT_LEVEL_NOTICE_GEMINI}

# ──────────────────────────────────────────────────────────────────────────────
# 2. Semantic Scholar 논문 검색 (prior research 검색 + 캐시, 2023년 이하)
# ──────────────────────────────────────────────────────────────────────────────
SEARCH_LIMIT = 20
SEARCH_FIELDS = "title,year,abstract,citationCount,url"
SEARCH_YEAR = "-2023"          # 2020~2023 특허 기반 컨셉이므로 2023년 이하 논문만 검색(시간적 누출 방지)
_SEARCH_CACHE = {}

def search_academic_resources(query: str) -> str:
    """Semantic Scholar 에서 query 관련 2023년 이하 논문(prior research)을 검색해 요약 문자열로 반환."""
    if query in _SEARCH_CACHE:
        return _SEARCH_CACHE[query]
    url = "https://api.semanticscholar.org/graph/v1/paper/search"
    try:
        resp = requests.get(url, params={"query": query, "limit": SEARCH_LIMIT,
                                         "offset": 0, "fields": SEARCH_FIELDS,
                                         "year": SEARCH_YEAR}, timeout=30)
        if resp.status_code != 200:
            out = f"검색 결과 없음 (status {resp.status_code})."
            _SEARCH_CACHE[query] = out
            return out
        papers = resp.json().get("data", []) or []
        # 방어적 필터: API year 파라미터가 무시되는 경우 대비, 2023년 초과 논문 제거
        papers = [p for p in papers
                  if not (isinstance(p.get("year"), int) and p["year"] > 2023)]
        if not papers:
            _SEARCH_CACHE[query] = "검색 결과 없음."
            return "검색 결과 없음."
        papers.sort(key=lambda p: p.get("citationCount", 0) or 0, reverse=True)
        lines = []
        for i, p in enumerate(papers[:15], 1):
            lines.append(f"[{i}] ({p.get('year','N/A')}, cited {p.get('citationCount',0) or 0}x) "
                         f"{p.get('title','N/A')}\n    Abstract: {p.get('abstract') or '(n/a)'}\n")
        out = f"총 {len(papers)}편 검색됨 (인용수 내림차순, {SEARCH_YEAR})\n" + "\n".join(lines)
        _SEARCH_CACHE[query] = out
        return out
    except Exception as e:
        return f"검색 중 오류: {e}"

# ──────────────────────────────────────────────────────────────────────────────
# 3. 프롬프트 (13.py expert_evaluate_node 와 동일 + JSON 출력 지시)
# ──────────────────────────────────────────────────────────────────────────────
def build_prompt(tech_text: str, ckey: str, tool_context: str, notice: str = None) -> str:
    notice = CONCEPT_LEVEL_NOTICE if notice is None else notice
    return f"""You are a {GLOBAL_PERSONA}.
{notice}

Criterion: [{ckey}]
Definition: {CRITERIA[ckey]}

<technology_concept>
{tech_text}
</technology_concept>

<search_context>
{tool_context if tool_context else "No external data yet."}
</search_context>

Instructions:
1. Evaluate whether this CONCEPT logically meets the criterion.
2. Base your judgment on the idea's theoretical direction, NOT on execution maturity.
3. Output a strict 'YES' or 'NO' verdict.
   - Default to 'YES' if the conceptual logic is sound and no clear violations exist.
   - A 'NO' requires an explicit, concrete reason (e.g., contradicts physics,
     is a verbatim copy of existing work, or has no identifiable problem).
4. If you need to check prior-art patents before deciding
   (especially for Novelty & Differentiation), set "needs_more_info" to true
   and provide patent search keywords in "search_queries".

Return ONLY a JSON object with exactly these keys:
{{"thought_process": "<step-by-step reasoning>",
  "search_queries": ["<kw1>", "<kw2>"],
  "needs_more_info": <true|false>,
  "verdict": "YES" | "NO"}}"""

# ──────────────────────────────────────────────────────────────────────────────
# 4. 모델 호출 (provider 별 구조화 JSON 출력)
# ──────────────────────────────────────────────────────────────────────────────
_clients = {}

def _get_client(provider):
    if provider in _clients:
        return _clients[provider]
    if provider == "openai":
        from openai import OpenAI
        _clients[provider] = OpenAI(api_key=OPENAI_API_KEY)
    elif provider == "anthropic":
        import anthropic
        _clients[provider] = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)
    elif provider == "google":
        from google import genai
        _clients[provider] = genai.Client(api_key=GOOGLE_API_KEY)
    return _clients[provider]

VERDICT_TOOL = {
    "name": "submit_verdict",
    "description": "Submit the criterion evaluation verdict.",
    "input_schema": {
        "type": "object",
        "properties": {
            "thought_process": {"type": "string"},
            "search_queries": {"type": "array", "items": {"type": "string"}},
            "needs_more_info": {"type": "boolean"},
            "verdict": {"type": "string", "enum": ["YES", "NO"]},
        },
        "required": ["thought_process", "needs_more_info", "verdict"],
    },
}

def _extract_json(text: str) -> dict:
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text, flags=re.S)
    m = re.search(r"\{.*\}", text, re.S)
    if m:
        try:
            return json.loads(m.group(0))
        except Exception:
            pass
    v = "YES" if re.search(r"\bYES\b", text, re.I) else ("NO" if re.search(r"\bNO\b", text, re.I) else "NO")
    return {"thought_process": text[:500], "search_queries": [], "needs_more_info": False, "verdict": v}

# 영구 오류(재시도 무의미): 크레딧/인증/권한/모델없음. "quota"는 Gemini 분당한도 429에도
# 들어가므로 영구 목록에서 제외한다.
_PERMANENT = ("credit balance", "authentication", "invalid api key", "invalid x-api-key",
              "billing", "model not found")
# 일시 오류(백오프 후 재시도): 429/분당한도/서버 과부하 등
_RETRYABLE = ("429", "resource_exhausted", "rate limit", "rate_limit", "overloaded",
              "unavailable", "503", "502", "500", "timeout", "deadline")

# provider 별 최소 호출 간격(초) — Gemini 무료 티어 RPM 회피용
_MIN_INTERVAL = {"google": 7.0, "openai": 0.0, "anthropic": 0.0}
_LAST_CALL = {}
# 일시 오류가 연속으로 누적되면(=하드 일일한도 추정) 해당 provider를 빠르게 포기
_CONSEC_FAIL = {}
_BREAKER_LIMIT = 8

def _throttle(provider):
    iv = _MIN_INTERVAL.get(provider, 0.0)
    if iv <= 0:
        return
    dt = time.time() - _LAST_CALL.get(provider, 0.0)
    if dt < iv:
        time.sleep(iv - dt)
    _LAST_CALL[provider] = time.time()

def call_model(provider, model, prompt, temperature=0.2, max_retries=6):
    if _CONSEC_FAIL.get(provider, 0) >= _BREAKER_LIMIT:
        raise RuntimeError(f"{provider} circuit-breaker open (반복적 한도 초과로 중단)")
    last = None
    for attempt in range(max_retries):
        try:
            _throttle(provider)
            client = _get_client(provider)
            if provider == "openai":
                r = client.chat.completions.create(
                    model=model, temperature=temperature,
                    response_format={"type": "json_object"},
                    messages=[{"role": "user", "content": prompt}])
                result = _extract_json(r.choices[0].message.content)
            elif provider == "anthropic":
                # Opus 4.8 / Sonnet 5 는 temperature/top_p/top_k 를 거부(400)하므로 전달하지 않음
                r = client.messages.create(
                    model=model, max_tokens=1200,
                    tools=[VERDICT_TOOL],
                    tool_choice={"type": "tool", "name": "submit_verdict"},
                    messages=[{"role": "user", "content": prompt}])
                result = next((b.input for b in r.content
                               if getattr(b, "type", None) == "tool_use"),
                              _extract_json("".join(getattr(b, "text", "") for b in r.content)))
            elif provider == "google":
                from google.genai import types
                r = client.models.generate_content(
                    model=model, contents=prompt,
                    config=types.GenerateContentConfig(
                        temperature=temperature, response_mime_type="application/json"))
                result = _extract_json(r.text or "")
            else:
                raise ValueError(f"unknown provider {provider}")
            _CONSEC_FAIL[provider] = 0      # 성공 시 연속 실패 카운터 리셋
            return result
        except Exception as e:
            last = e
            msg = str(e).lower()
            is_retryable = any(r in msg for r in _RETRYABLE)
            if any(p in msg for p in _PERMANENT) and not is_retryable:
                raise            # 영구 오류(크레딧/인증/모델없음)는 재시도하지 않음
            if is_retryable:
                _CONSEC_FAIL[provider] = _CONSEC_FAIL.get(provider, 0) + 1
                if _CONSEC_FAIL[provider] >= _BREAKER_LIMIT:
                    raise
                time.sleep(20)   # 분당 한도 회복 대기
            else:
                time.sleep(2 + attempt * 2)
    raise last

# ──────────────────────────────────────────────────────────────────────────────
# 5. Agent 1-round reflection (evaluate → search → re-evaluate)
# ──────────────────────────────────────────────────────────────────────────────
def judge_criterion(provider, model, tech_text, ckey, notice=None):
    res = call_model(provider, model, build_prompt(tech_text, ckey, "", notice))
    used_search = False
    if res.get("needs_more_info") and res.get("search_queries"):
        used_search = True
        ctx = ""
        for q in (res["search_queries"] or [])[:2]:
            ctx += f"\n[Query: '{q}']\n{search_academic_resources(q)}"
            time.sleep(0.3)
        res = call_model(provider, model, build_prompt(tech_text, ckey, ctx, notice))
    verdict = str(res.get("verdict", "NO")).strip().upper()
    verdict = "YES" if verdict == "YES" else "NO"
    return {"verdict": verdict,
            "rationale": str(res.get("thought_process", ""))[:1500],
            "used_search": used_search}

# ──────────────────────────────────────────────────────────────────────────────
# 6. 실행
# ──────────────────────────────────────────────────────────────────────────────
def load_examples():
    # prospective_nodes_with_openai_similarity.xlsx 의 55개 technology concept 전체를 평가한다.
    # Phase: 문법 교정본(decoded_text=corrected)을 judge 입력으로 사용
    df = pd.read_excel(f"{D}/data/prospective_nodes_grammar_corrected.xlsx")
    df = df[df["decoded_text"].notna()].reset_index(drop=True)
    return [{"label": f"EX{i+1}", "node_id": df.loc[i, "node_id"],
             "text": df.loc[i, "decoded_text"]} for i in range(len(df))]

def main():
    examples = load_examples()
    print(f"📊 검증 대상: {len(examples)}개 텍스트 × {len(CRITERIA)}개 기준 × {len(MODELS)}개 모델\n")

    per_model_rows = {m: [] for m in MODELS}     # 모델별 상세 결과
    model_status = {}                            # 모델별 가용 여부

    n = len(examples)
    # JUDGE_RESUME=1 이면, 이미 완결된(=55행·ERROR 없음) 모델 결과 파일은 재판정 없이 재사용한다.
    resume = os.environ.get("JUDGE_RESUME") == "1"

    def load_done(mname):
        """완결된 기존 결과를 불러온다. 불완전하면 None."""
        p = f"{D}/data/agentjudge_3model_{mname.lower()}.xlsx"
        if not (resume and os.path.exists(p)):
            return None
        try:
            df = pd.read_excel(p)
        except Exception:
            return None
        need = [f"{c} | verdict" for c in CRIT_KEYS]
        if len(df) != n or any(c not in df.columns for c in need):
            return None
        if (df[need] == "ERROR").any().any():
            return None
        # 저장된 판정이 '현재 입력 텍스트'로 매겨진 것인지 검증 (텍스트가 바뀌면 재판정)
        if "text" not in df.columns:
            return None
        stored = df["text"].astype(str).str.strip().tolist()
        current = [str(ex["text"]).strip() for ex in examples]
        if stored != current:
            return None
        return df.to_dict("records")

    for mname, cfg in MODELS.items():
        print(f"{'='*70}\n🤖 MODEL: {mname}  ({cfg['model']})\n{'='*70}")
        done = load_done(mname)
        if done is not None:
            per_model_rows[mname] = done
            model_status[mname] = "OK (resumed)"
            yes_n = sum(1 for r in done if r.get("overall_verdict") == "YES")
            print(f"⏭️  기존 완결 결과 재사용 (재판정 생략) — overall YES {yes_n}/{n}\n")
            continue
        ok = True
        for idx, ex in enumerate(examples, 1):
            row = {"label": ex["label"], "node_id": ex["node_id"],
                   "text": ex["text"]}
            yes = 0
            for ckey in CRIT_KEYS:
                try:
                    out = judge_criterion(cfg["provider"], cfg["model"], ex["text"], ckey,
                                          NOTICE_BY_MODEL.get(mname))
                    row[f"{ckey} | verdict"] = out["verdict"]
                    row[f"{ckey} | rationale"] = out["rationale"]
                    if out["verdict"] == "YES":
                        yes += 1
                    print(f"  [{ex['label']} {idx}/{n}] {ckey[:34]:34s} -> {out['verdict']}"
                          f"{' (searched)' if out['used_search'] else ''}")
                except Exception as e:
                    row[f"{ckey} | verdict"] = "ERROR"
                    row[f"{ckey} | rationale"] = str(e)[:300]
                    ok = False
                    print(f"  [{ex['label']} {idx}/{n}] {ckey[:34]:34s} -> ERROR: {str(e)[:90]}")
            row["total_yes"] = yes
            # 종합 YES/NO: 모든 기준이 YES(만장일치)일 때만 YES
            row["overall_verdict"] = "YES" if yes == len(CRIT_KEYS) else "NO"
            per_model_rows[mname].append(row)
            print(f"    => {ex['label']} total_yes={yes}/{len(CRIT_KEYS)}  overall={row['overall_verdict']}\n")
        model_status[mname] = "OK" if ok else "PARTIAL/UNAVAILABLE"
        # 모델별 결과를 완료 즉시 저장(긴 실행 중 크래시/한도 대비, 완료된 모델은 보존)
        out_path = f"{D}/data/agentjudge_3model_{mname.lower()}.xlsx"
        pd.DataFrame(per_model_rows[mname]).to_excel(out_path, index=False)
        print(f"💾 {mname} 상세 결과 저장: {out_path}  [{model_status[mname]}]\n")

    # ── 종합 표 (overall) ────────────────────────────────────────────
    combined = []
    for i, ex in enumerate(examples):
        rec = {"label": ex["label"], "node_id": ex["node_id"],
               "text_snippet": str(ex["text"])[:90] + "..."}
        overalls = []
        for mname in MODELS:
            r = per_model_rows[mname][i]
            rec[f"{mname}_total_yes"] = r["total_yes"]
            rec[f"{mname}_overall"] = r["overall_verdict"]
            if r["overall_verdict"] in ("YES", "NO"):
                overalls.append(r["overall_verdict"])
        yc = overalls.count("YES")
        rec["models_voted"] = len(overalls)
        rec["YES_votes"] = yc
        rec["consensus_overall"] = ("YES" if yc * 2 > len(overalls) else "NO") if overalls else "N/A"
        combined.append(rec)
    combined_df = pd.DataFrame(combined)

    # ── 종합 표 (기준별 3모델 비교) ──────────────────────────────────
    per_crit = []
    for i, ex in enumerate(examples):
        for ckey in CRIT_KEYS:
            rec = {"label": ex["label"], "criterion": ckey}
            votes = []
            for mname in MODELS:
                v = per_model_rows[mname][i].get(f"{ckey} | verdict", "ERROR")
                rec[mname] = v
                if v in ("YES", "NO"):
                    votes.append(v)
            yc = votes.count("YES")
            rec["majority"] = ("YES" if yc * 2 > len(votes) else "NO") if votes else "N/A"
            per_crit.append(rec)
    per_crit_df = pd.DataFrame(per_crit)

    combined_path = f"{D}/data/agentjudge_3model_combined.xlsx"
    with pd.ExcelWriter(combined_path) as xw:
        combined_df.to_excel(xw, sheet_name="overall", index=False)
        per_crit_df.to_excel(xw, sheet_name="by_criterion", index=False)
    print(f"\n🎉 종합 표 저장: {combined_path}")
    print(f"   모델 상태: {model_status}")

    print("\n================ OVERALL 종합 ================")
    print(combined_df.to_string(index=False))
    print("\n================ 기준별 비교 ================")
    print(per_crit_df.to_string(index=False))

if __name__ == "__main__":
    main()
