"""
20. 생성된 technology concept 문법 교정 (아이디어·길이 유지, 문법/오탈자만)
============================================================================
prospective_nodes_with_openai_similarity.xlsx 의 decoded_text 55개를
gpt-5.4-mini 로 '문법/오탈자/깨진 표현'만 교정한다.

원칙(프롬프트로 강제):
 - 기술 내용·아이디어 불변, 길이 유지, 내용 추가/삭제/발명 금지
 - 후속특허 등 외부 문서 참조 금지 (blind) → 유사도 인위적 부풀림 방지
 - 잘린 문장은 없는 내용을 지어내지 말고 최소한으로만 정리

출력: data/prospective_nodes_grammar_corrected.xlsx
 (node_id, decoded_text_original, decoded_text[=corrected], len_orig, len_corr, len_ratio)
"""
# Source file in the working repository: code/20.grammar_correct_concepts.py
# Release edits: API keys removed (environment variables), paths made repository-relative.
# --- public-release addition: repository root for relative paths ------------
import os as _os
_ROOT = _os.path.abspath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '..', '..'))
# ---------------------------------------------------------------------------
import os, re, time
import pandas as pd
from openai import OpenAI

D = _ROOT
MODEL = "gpt-5.4-mini"
client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

SYSTEM = (
    "You are a copy editor for technical text. You will be given a technology "
    "concept description that was produced by an automated text-generation system and may contain "
    "grammatical errors, misspellings, awkward or repetitive phrasing, or abrupt truncation. "
    "Your ONLY job is to correct grammar, spelling, punctuation, and clearly "
    "broken/duplicated wording so the text reads cleanly.\n"
    "STRICT RULES:\n"
    "1. Preserve the original meaning and ALL technical content exactly. Do not add, remove, or "
    "invent any information, entities, or claims.\n"
    "2. Keep the length approximately the same (do not expand or summarize).\n"
    "3. Do NOT complete or fill in truncated/incomplete sentences with invented content; only smooth "
    "them minimally (e.g., trim a dangling fragment) without adding new ideas.\n"
    "4. Do NOT change or standardize technical terminology, and do NOT reference or assume any "
    "external document or patent.\n"
    "5. Return ONLY the corrected text, with no preamble, quotes, or commentary."
)

def correct(text: str) -> str:
    for attempt in range(4):
        try:
            r = client.chat.completions.create(
                model=MODEL, temperature=0,
                messages=[{"role": "system", "content": SYSTEM},
                          {"role": "user", "content": text}])
            return r.choices[0].message.content.strip()
        except Exception as e:
            if attempt == 3:
                raise
            time.sleep(3 + attempt * 3)

def main():
    df = pd.read_excel(f"{D}/data/prospective_nodes_with_openai_similarity.xlsx")
    df = df[df["decoded_text"].notna()].reset_index(drop=True)
    rows = []
    for i, r in df.iterrows():
        orig = str(r["decoded_text"])
        corr = correct(orig)
        rows.append({"node_id": r["node_id"], "decoded_text_original": orig,
                     "decoded_text": corr, "len_orig": len(orig), "len_corr": len(corr)})
        print(f"[{i+1}/{len(df)}] {r['node_id']}  len {len(orig)}->{len(corr)}  ratio {len(corr)/max(1,len(orig)):.2f}")
    out = pd.DataFrame(rows)
    out["len_ratio"] = (out["len_corr"] / out["len_orig"]).round(3)
    path = f"{D}/data/prospective_nodes_grammar_corrected.xlsx"
    out.to_excel(path, index=False)
    print(f"\n[OK] saved {path}  rows={len(out)}  "
          f"len_ratio mean={out['len_ratio'].mean():.3f} min={out['len_ratio'].min():.3f} max={out['len_ratio'].max():.3f}")

if __name__ == "__main__":
    main()
