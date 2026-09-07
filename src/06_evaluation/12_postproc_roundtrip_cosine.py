"""
23. 후처리(문법 교정) 영향 분석 - (3) 라운드트립 일관성
============================================================================
타깃 임베딩 : prospective node 의 projected text embedding (deep MLP 출력, ada-002 공간)
              = vec2text 가 역변환하려 했던 원 타깃
비교 대상   : raw(vec2text 원 출력) / post(v1 문법 교정) 텍스트를
              text-embedding-ada-002 로 재임베딩한 벡터

cos(target, emb(text)) 를 raw / post 각각 계산하여 후처리가 라운드트립
일관성을 훼손하지 않았는지 확인한다.

출력: data/postproc_roundtrip_cosine.csv
"""
# Source file in the working repository: code/23.postproc_roundtrip_cosine.py
# Release edits: API keys removed (environment variables), paths made repository-relative.
# --- public-release addition: repository root for relative paths ------------
import os as _os
_ROOT = _os.path.abspath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '..', '..'))
# ---------------------------------------------------------------------------
import os
import ast
import numpy as np
import pandas as pd
from openai import OpenAI

D = _ROOT
EMB_MODEL = "text-embedding-ada-002"
client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])


def embed(texts):
    out = []
    for i in range(0, len(texts), 32):
        chunk = [t.replace("\n", " ") for t in texts[i:i + 32]]
        r = client.embeddings.create(model=EMB_MODEL, input=chunk)
        out.extend([d.embedding for d in sorted(r.data, key=lambda x: x.index)])
    return np.array(out, dtype=np.float64)


def cos(A, B):
    A = A / np.linalg.norm(A, axis=1, keepdims=True)
    B = B / np.linalg.norm(B, axis=1, keepdims=True)
    return (A * B).sum(1)


def main():
    dec = pd.read_csv(f"{D}/data/final_decoded_results.csv")          # id, projected_text_embedding, round_trip_cosine_sim
    grm = pd.read_excel(f"{D}/data/prospective_nodes_grammar_corrected.xlsx")
    df = grm.merge(dec[["id", "projected_text_embedding", "round_trip_cosine_sim"]],
                   left_on="node_id", right_on="id", how="left", validate="1:1")
    assert df["projected_text_embedding"].notna().all(), "타깃 임베딩 매칭 실패"

    tgt = np.array([ast.literal_eval(s) for s in df["projected_text_embedding"]], dtype=np.float64)
    E_raw = embed(df["decoded_text_original"].astype(str).tolist())
    E_post = embed(df["decoded_text"].astype(str).tolist())

    out = pd.DataFrame({
        "node_id": df["node_id"],
        "cos_target_raw": cos(tgt, E_raw).round(6),
        "cos_target_post": cos(tgt, E_post).round(6),
        "cos_raw_post": cos(E_raw, E_post).round(6),
        "stored_round_trip_raw": df["round_trip_cosine_sim"].round(6),
    })
    out["delta_post_minus_raw"] = (out.cos_target_post - out.cos_target_raw).round(6)
    path = f"{D}/data/postproc_roundtrip_cosine.csv"
    out.to_csv(path, index=False)

    print(out.to_string(index=False))
    print("\n=== 요약 ===")
    for c in ["cos_target_raw", "cos_target_post", "delta_post_minus_raw", "cos_raw_post"]:
        s = out[c]
        print(f"{c:24s} mean={s.mean():.4f} sd={s.std():.4f} min={s.min():.4f} "
              f"median={s.median():.4f} max={s.max():.4f}")
    print(f"\npost > raw : {(out.delta_post_minus_raw > 0).sum()}/{len(out)}, "
          f"post < raw : {(out.delta_post_minus_raw < 0).sum()}/{len(out)}")
    # 재현성 확인: 저장된 round_trip_cosine_sim(스크립트 11) 과 재계산값 일치 여부
    d = (out.cos_target_raw - out.stored_round_trip_raw).abs()
    print(f"저장된 raw round-trip 과의 최대 절대차 = {d.max():.6f} (재현성 확인)")
    print(f"[OK] saved {path}")


if __name__ == "__main__":
    main()
