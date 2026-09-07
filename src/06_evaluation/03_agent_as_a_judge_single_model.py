# Source file in the working repository: code/13.agent-as-a-Judge.py
# Release edits: API keys removed (environment variables), paths made repository-relative.
# --- public-release addition: repository root for relative paths ------------
import os as _os
_ROOT = _os.path.abspath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '..', '..'))
# ---------------------------------------------------------------------------
import os, time
import pandas as pd
from tqdm import tqdm
import requests
from typing import List, Literal, TypedDict, Optional
from pydantic import BaseModel, Field

from openai import OpenAI
from langgraph.graph import StateGraph, END

# ══════════════════════════════════════════════════════════════════════════════
# 🔑 API Keys
# ══════════════════════════════════════════════════════════════════════════════
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY
client = OpenAI(api_key=OPENAI_API_KEY)

# ══════════════════════════════════════════════════════════════════════════════
# 1. Schema Definitions
# ══════════════════════════════════════════════════════════════════════════════

class AgentVerdict(BaseModel):
    thought_process: str = Field(
        description="초기 기술 컨셉이라는 점을 전제로 한 단계별 이론적 타당성 분석"
    )
    search_queries_used: List[str] = Field(
        description="검증을 위해 검색해 본 키워드 (필요시)"
    )
    citations: str = Field(
        description="근거가 된 주요 선행 연구 또는 개념"
    )
    needs_more_info: bool = Field(
        description="검색을 통한 추가 정보가 반드시 필요한지 여부 (True/False)"
    )
    verdict: Literal["YES", "NO"] = Field(
        description="해당 평가 기준의 충족 여부 (YES 또는 NO)"
    )


class TechnicalCriticism(BaseModel):
    theoretical_flaws: List[str] = Field(
        description="순수 이론적·논리적 모순점 (메커니즘 부재, 데이터 부재는 해당하지 않음)"
    )
    conceptual_coherence: int = Field(
        ge=1, le=5,
        description="컨셉 수준 논리 정합성 점수 (1~5)"
    )


class FinalTechVerdict(BaseModel):
    final_synthesis: str = Field(
        description="아이디어의 잠재력을 종합한 최종 요약"
    )
    final_verdict: Literal["HIGH_POTENTIAL", "MODERATE_POTENTIAL", "NEEDS_REVISION"] = Field(
        description="최종 판정"
    )


# ══════════════════════════════════════════════════════════════════════════════
# 2. Evaluation Criteria — Technology Concept Level
# ══════════════════════════════════════════════════════════════════════════════
# 평가 대상은 "기술 컨셉(Technology Concept)"입니다.
# 컨셉 수준에서는 구체적 메커니즘 설계, 실험 데이터, 프로토타입이 존재하지 않는 것이
# 정상적인 상태이므로, 이를 감점 사유로 삼지 않습니다.
# 대신 아이디어 자체의 논리적 방향성과 이론적 잠재력을 평가합니다.
# ══════════════════════════════════════════════════════════════════════════════

CRITERIA = {
    "Originality & Novelty": (
        "Does the concept propose a genuinely new angle, combination, or framing "
        "that is not a trivial restatement of existing work? "
        "Evaluate by comparing the core proposition against the landscape of prior "
        "research retrieved via search. A concept may combine known elements in a "
        "novel way — this still counts as original."
    ),
    "Scientific & Logical Consistency": (
        "Are the concept's foundational assumptions compatible with established "
        "scientific principles? Is the internal reasoning free of logical "
        "contradictions? NOTE: The absence of a detailed mechanism or mathematical "
        "formulation is EXPECTED at concept level and must NOT be treated as a flaw. "
        "Focus only on whether the stated logic violates known laws or contradicts itself."
    ),
    "Problem–Concept Alignment": (
        "Is the problem the concept addresses clearly identified and significant? "
        "Does the proposed conceptual direction offer a plausible path toward "
        "addressing that problem? A concept does not need to fully solve the problem; "
        "it only needs to point in a defensible direction."
    ),
    "Research & Development Potential": (
        "Does the concept open meaningful avenues for future research or engineering "
        "exploration? Could it inspire follow-up studies, prototyping, or theoretical "
        "extensions? Evaluate the fertility of the idea, not whether a roadmap exists."
    ),
    "Differentiated Value Proposition": (
        "Does the concept articulate a clear advantage or unique benefit compared to "
        "existing or alternative approaches? The advantage may be theoretical "
        "(e.g., a simpler framing, broader applicability) rather than empirically proven."
    ),
}

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

# ══════════════════════════════════════════════════════════════════════════════
# 3. Search Tools — Semantic Scholar (50 papers, full abstract)
# ══════════════════════════════════════════════════════════════════════════════

SEARCH_LIMIT = 50  # 한 번에 가져올 논문 수
SEARCH_FIELDS = "title,year,abstract,citationCount,url"
SEARCH_YEAR = "-2023"  # 2020~2023 특허 기반 컨셉이므로 2023년 이하 논문만 검색 (시간적 누출 방지)

def search_academic_resources(query: str) -> str:
    """
    Semantic Scholar API로 최대 50편의 논문을 검색하고,
    title, year, abstract 전문, citation count를 반환합니다.
    """
    url = "https://api.semanticscholar.org/graph/v1/paper/search"
    all_papers = []
    offset = 0
    remaining = SEARCH_LIMIT

    try:
        # Semantic Scholar API는 한 번에 최대 100편까지 지원하므로
        # 50편은 단일 요청으로 충분하지만, 안전을 위해 페이징 구조 유지
        while remaining > 0:
            batch_size = min(remaining, 100)
            resp = requests.get(
                url,
                params={
                    "query": query,
                    "limit": batch_size,
                    "offset": offset,
                    "fields": SEARCH_FIELDS,
                    "year": SEARCH_YEAR,
                },
                timeout=30,
            )
            if resp.status_code != 200:
                break
            data = resp.json()
            papers = data.get("data", [])
            if not papers:
                break
            all_papers.extend(papers)
            offset += len(papers)
            remaining -= len(papers)
            # API rate limit 방지
            time.sleep(0.5)

        # 방어적 필터: API year 파라미터가 무시되는 경우 대비, 2023년 초과 논문 제거
        all_papers = [p for p in all_papers
                      if not (isinstance(p.get("year"), int) and p["year"] > 2023)]

        if not all_papers:
            return "검색 결과 없음."

        # 인용수 기준 내림차순 정렬 (영향력 높은 논문 우선)
        all_papers.sort(key=lambda p: p.get("citationCount", 0) or 0, reverse=True)

        lines = []
        for i, p in enumerate(all_papers, 1):
            title = p.get("title", "N/A")
            year = p.get("year", "N/A")
            cites = p.get("citationCount", 0) or 0
            abstract = p.get("abstract") or "(abstract not available)"
            paper_url = p.get("url", "")
            lines.append(
                f"[{i}] ({year}, cited {cites}x) {title}\n"
                f"    URL: {paper_url}\n"
                f"    Abstract: {abstract}\n"
            )

        header = f"총 {len(all_papers)}편의 논문 검색됨 (인용수 내림차순)\n{'='*60}\n"
        return header + "\n".join(lines)

    except requests.exceptions.Timeout:
        return "검색 시간 초과 (timeout)."
    except Exception as e:
        return f"검색 중 오류 발생: {str(e)}"


# ══════════════════════════════════════════════════════════════════════════════
# 4. LangGraph Workflow — Criterion-level Evaluation
# ══════════════════════════════════════════════════════════════════════════════

class EvalState(TypedDict):
    tech_text: str
    criterion_key: str
    tool_context: str
    reflection_count: int
    result: Optional[AgentVerdict]


def expert_evaluate_node(state: EvalState) -> dict:
    ckey = state["criterion_key"]
    c_desc = CRITERIA[ckey]

    prompt = f"""You are a {GLOBAL_PERSONA}.
{CONCEPT_LEVEL_NOTICE}

Criterion: [{ckey}]
Definition: {c_desc}

<technology_concept>
{state['tech_text']}
</technology_concept>

<search_context>
{state['tool_context'] if state['tool_context'] else "No external data yet."}
</search_context>

Instructions:
1. Evaluate whether this CONCEPT logically meets the criterion.
2. Base your judgment on the idea's theoretical direction, NOT on execution maturity.
3. Output a strict 'YES' or 'NO' verdict.
   - Default to 'YES' if the conceptual logic is sound and no clear violations exist.
   - A 'NO' requires an explicit, concrete reason (e.g., contradicts physics,
     is a verbatim copy of existing work, or has no identifiable problem).
4. If you need to check the academic literature before deciding
   (especially for Originality & Novelty), set 'needs_more_info' to True
   and provide search keywords in 'search_queries_used'.
"""
    response = client.beta.chat.completions.parse(
        model="gpt-5.4-mini",
        messages=[{"role": "user", "content": prompt}],
        response_format=AgentVerdict,
        temperature=0.2,
    )
    return {"result": response.choices[0].message.parsed}


def search_node(state: EvalState) -> dict:
    queries = state["result"].search_queries_used if state["result"].search_queries_used else [state["tech_text"][:60]]
    
    # 여러 검색 키워드가 있으면 모두 검색
    all_results = []
    for q in queries[:3]:  # 최대 3개 쿼리까지
        result = search_academic_resources(q)
        all_results.append(f"\n[Query: '{q}']\n{result}")
        time.sleep(1)  # API rate limit
    
    combined = "\n".join(all_results)
    
    return {
        "tool_context": state["tool_context"] + combined,
        "reflection_count": state["reflection_count"] + 1,
    }


def should_continue(state: EvalState):
    if state["reflection_count"] >= 1 or not state["result"].needs_more_info:
        return END
    return "search"


builder = StateGraph(EvalState)
builder.add_node("evaluate", expert_evaluate_node)
builder.add_node("search", search_node)
builder.set_entry_point("evaluate")
builder.add_conditional_edges("evaluate", should_continue, {"search": "search", END: END})
builder.add_edge("search", "evaluate")
eval_app = builder.compile()


# ══════════════════════════════════════════════════════════════════════════════
# 5. Idea Critic & Final Synthesis
# ══════════════════════════════════════════════════════════════════════════════

def idea_critic_node(tech_text: str, results: List[AgentVerdict]) -> TechnicalCriticism:
    prompt = f"""You are a Concept-Level Idea Critic.
{CONCEPT_LEVEL_NOTICE}

<technology_concept>
{tech_text}
</technology_concept>

Your job is to identify ONLY genuine theoretical weaknesses.
The following are NOT valid criticisms at concept level:
  - "No detailed mechanism is provided"
  - "No experimental validation"
  - "Lacks quantitative analysis"
  - "No prototype or implementation"
  - "Needs more technical specifics"

Valid criticisms include:
  - The concept contradicts established scientific principles
  - The internal logic is self-contradictory
  - The concept is indistinguishable from existing well-known approaches (no novelty)
  - The claimed benefit does not logically follow from the concept
  - The problem being addressed is mischaracterized or trivial

Score `conceptual_coherence` from 1 to 5 (HIGHER = BETTER):
  1: Fundamentally incoherent — contradicts basic science or is pure pseudoscience.
  2: Contains significant logical contradictions within the concept itself.
  3: Logically consistent but the core proposition is vague, generic,
     or indistinguishable from existing common approaches.
  4: Sound conceptual logic with a clear and defensible direction.
     Minor ambiguities may exist but do not undermine the core idea.
  5: Exceptionally coherent — the conceptual framework is tight,
     internally consistent, and clearly articulated.

REMINDER: Absence of mechanism details is NORMAL and must score ≥ 3
if the conceptual direction itself is logically sound.
"""
    response = client.beta.chat.completions.parse(
        model="gpt-5.4-mini",
        messages=[{"role": "user", "content": prompt}],
        response_format=TechnicalCriticism,
        temperature=0.3,
    )
    return response.choices[0].message.parsed


def final_synthesis_node(
    tech_text: str,
    results: List[AgentVerdict],
    critic: TechnicalCriticism,
) -> FinalTechVerdict:
    yes_count = sum(1 for r in results if r.verdict == "YES")

    prompt = f"""You are a Meta-Judge synthesizing the evaluation of a TECHNOLOGY CONCEPT.
{CONCEPT_LEVEL_NOTICE}

═══ Evaluation Summary ═══
Total YES Verdicts: {yes_count} / 5
Conceptual Coherence Score: {critic.conceptual_coherence} / 5 (Higher = Better)
Identified Theoretical Flaws: {'; '.join(critic.theoretical_flaws) if critic.theoretical_flaws else 'None'}

═══ Decision Rules ═══
- HIGH_POTENTIAL:     ≥ 4 YES verdicts AND conceptual coherence ≥ 4
- MODERATE_POTENTIAL: 2–3 YES verdicts OR conceptual coherence = 3
- NEEDS_REVISION:     ≤ 1 YES verdict  OR conceptual coherence ≤ 2

Provide:
1. A concise final_synthesis summarizing the concept's strengths and areas for improvement.
2. The final_verdict based strictly on the rules above.
"""
    response = client.beta.chat.completions.parse(
        model="gpt-5.4-mini",
        messages=[{"role": "user", "content": prompt}],
        response_format=FinalTechVerdict,
        temperature=0.1,
    )
    return response.choices[0].message.parsed


# ══════════════════════════════════════════════════════════════════════════════
# 6. Main Execution
# ══════════════════════════════════════════════════════════════════════════════

def main():
    print("🚀 Starting Agent-as-a-Judge v8.0: Technology Concept Evaluation")
    print("   - 5 criteria (concept-level adapted)")
    print("   - 50 papers per search, full abstracts")
    print("   - Concept-level notice injected in all prompts\n")

    INPUT_CSV = _ROOT + "/data/final_decoded_results_with_nn.csv"
    OUTPUT_CSV = _ROOT + "/data/evaluation_results_v8.csv"

    try:
        df = pd.read_csv(INPUT_CSV)
        concepts = df["decoded_text"].dropna().tolist()
        print(f"📊 총 {len(concepts)}개의 기술 컨셉을 평가합니다.\n")
    except Exception as e:
        print(f"❌ CSV 로드 실패: {e}")
        return

    final_results = []

    for idx, concept in enumerate(concepts):
        display_text = concept[:50].replace("\n", " ")
        print(f"[{idx+1}/{len(concepts)}] 💡 평가 중: {display_text}...")

        try:
            step_results = []
            for ckey in CRITERIA.keys():
                res = eval_app.invoke(
                    {
                        "tech_text": concept,
                        "criterion_key": ckey,
                        "tool_context": "",
                        "reflection_count": 0,
                        "result": None,
                    }
                )
                step_results.append(res["result"])

            critic_out = idea_critic_node(concept, step_results)
            final_out = final_synthesis_node(concept, step_results, critic_out)

            yes_count = sum(1 for r in step_results if r.verdict == "YES")
            print(
                f"  ✅ YES: {yes_count}/5 | "
                f"Coherence: {critic_out.conceptual_coherence}/5 | "
                f"Verdict: {final_out.final_verdict}"
            )

            result_row = {
                "original_text": concept,
                "final_verdict": final_out.final_verdict,
                "total_yes_count": yes_count,
                "conceptual_coherence": critic_out.conceptual_coherence,
                "theoretical_flaws": " | ".join(critic_out.theoretical_flaws),
                "final_synthesis": final_out.final_synthesis,
            }

            for i, ckey in enumerate(CRITERIA.keys()):
                result_row[f"{ckey}_verdict"] = step_results[i].verdict
                result_row[f"{ckey}_rationale"] = step_results[i].thought_process

            final_results.append(result_row)
            time.sleep(1)

        except Exception as e:
            print(f"  ❌ 에러 발생: {e}")
            final_results.append({
                "original_text": concept,
                "final_verdict": "ERROR",
                "total_yes_count": -1,
                "conceptual_coherence": -1,
                "theoretical_flaws": str(e),
                "final_synthesis": "",
            })

    res_df = pd.DataFrame(final_results)
    res_df.to_csv(OUTPUT_CSV, index=False, encoding="utf-8-sig", escapechar="\\")
    print(f"\n🎉 전체 평가 완료! 결과: {OUTPUT_CSV}")


if __name__ == "__main__":
    main()