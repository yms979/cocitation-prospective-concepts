"""
21. 후처리(문법 교정) 영향 분석 - (2) 토큰 수준 편집거리 분포
============================================================================
raw = decoded_text_original (vec2text 원 출력)
post = decoded_text (v1 문법 교정 결과)

토큰화: 소문자 + 단어/숫자 단위 정규식 (\\w+)
편집거리: 토큰 시퀀스 Levenshtein (ins/del/sub 비용 1)
정규화: distance / max(len_raw_tok, len_post_tok)

출력: data/postproc_edit_distance.csv  (노드별)
"""
# Source file in the working repository: code/21.postproc_edit_distance.py
# Release edits: API keys removed (environment variables), paths made repository-relative.
# --- public-release addition: repository root for relative paths ------------
import os as _os
_ROOT = _os.path.abspath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '..', '..'))
# ---------------------------------------------------------------------------
import re
import pandas as pd

D = _ROOT
TOK = re.compile(r"[A-Za-z0-9]+(?:[-'][A-Za-z0-9]+)*")


def tokens(t: str):
    return TOK.findall(str(t).lower())


def levenshtein(a, b):
    """토큰 시퀀스 Levenshtein + 연산별 카운트(백트레이스)."""
    n, m = len(a), len(b)
    dp = [[0] * (m + 1) for _ in range(n + 1)]
    for i in range(n + 1):
        dp[i][0] = i
    for j in range(m + 1):
        dp[0][j] = j
    for i in range(1, n + 1):
        ai = a[i - 1]
        for j in range(1, m + 1):
            cost = 0 if ai == b[j - 1] else 1
            dp[i][j] = min(dp[i - 1][j] + 1, dp[i][j - 1] + 1, dp[i - 1][j - 1] + cost)
    # 백트레이스로 sub/ins/del 분해
    i, j, sub = n, m, 0
    ins = dele = 0
    while i > 0 or j > 0:
        if i > 0 and j > 0:
            cost = 0 if a[i - 1] == b[j - 1] else 1
            if dp[i][j] == dp[i - 1][j - 1] + cost:
                sub += cost
                i, j = i - 1, j - 1
                continue
        if i > 0 and dp[i][j] == dp[i - 1][j] + 1:
            dele += 1
            i -= 1
            continue
        j -= 1
        ins += 1
    return dp[n][m], sub, ins, dele


def main():
    df = pd.read_excel(f"{D}/data/prospective_nodes_grammar_corrected.xlsx")
    rows = []
    for _, r in df.iterrows():
        a, b = tokens(r["decoded_text_original"]), tokens(r["decoded_text"])
        dist, sub, ins, dele = levenshtein(a, b)
        denom = max(len(a), len(b), 1)
        rows.append({
            "node_id": r["node_id"],
            "n_tok_raw": len(a), "n_tok_post": len(b),
            "edit_distance": dist, "sub": sub, "ins": ins, "del": dele,
            "norm_edit_distance": round(dist / denom, 4),
            "identical": int(dist == 0),
        })
    out = pd.DataFrame(rows)
    path = f"{D}/data/postproc_edit_distance.csv"
    out.to_csv(path, index=False)

    q = out["edit_distance"].describe(percentiles=[.25, .5, .75, .9])
    print(out.to_string(index=False))
    print("\n=== edit_distance (tokens) ===")
    print(q.round(3).to_string())
    print("\n=== norm_edit_distance ===")
    print(out["norm_edit_distance"].describe(percentiles=[.25, .5, .75, .9]).round(4).to_string())
    print(f"\nidentical(변경 없음) = {out.identical.sum()}/{len(out)}")
    print(f"총 편집연산: sub={out['sub'].sum()} ins={out['ins'].sum()} del={out['del'].sum()}")
    print(f"[OK] saved {path}")


if __name__ == "__main__":
    main()
