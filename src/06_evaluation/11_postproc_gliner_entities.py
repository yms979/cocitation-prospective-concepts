"""
22. 후처리(문법 교정) 영향 분석 - (1) GLiNER 기술 개체 추출 & 신규 개체 출현율
============================================================================
raw  = decoded_text_original (vec2text 원 출력)
post = decoded_text (v1 문법 교정 결과)

GLiNER(urchade/gliner_medium-v2.1)로 양쪽에서 기술 개체를 추출한 뒤,
 - new  : post 에만 있는 개체 (= 후처리가 없던 내용을 만들어냈는가; 낮아야 함)
 - lost : raw 에만 있는 개체 (= 후처리가 내용을 지웠는가)
 - kept : 양쪽 공통
개체 매칭은 소문자·공백정규화·양끝 관사/구두점 제거 후 표면형 집합 기준(set).

출력: data/postproc_gliner_entities.csv (노드별 카운트 + 개체 목록)
"""
# Source file in the working repository: code/22.postproc_gliner_entities.py
# Release edits: API keys removed (environment variables), paths made repository-relative.
# --- public-release addition: repository root for relative paths ------------
import os as _os
_ROOT = _os.path.abspath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '..', '..'))
# ---------------------------------------------------------------------------
import os
import re
import pandas as pd

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

D = _ROOT
MODEL = "urchade/gliner_medium-v2.1"
THRESHOLD = 0.5
LABELS = [
    "technology", "device", "system", "component",
    "material", "chemical substance", "method", "measurement or parameter",
]

_ART = re.compile(r"^(a|an|the|said|such)\s+", re.I)


def norm(s: str) -> str:
    s = re.sub(r"\s+", " ", str(s).strip().lower())
    s = _ART.sub("", s)
    return s.strip(" .,;:()-")


def main():
    from gliner import GLiNER
    import torch

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    model = GLiNER.from_pretrained(MODEL).to(dev).eval()
    print(f"[model] {MODEL} on {dev}  labels={LABELS}  threshold={THRESHOLD}")

    df = pd.read_excel(f"{D}/data/prospective_nodes_grammar_corrected.xlsx")
    rows = []
    for i, r in df.iterrows():
        ents = {}
        for key, col in (("raw", "decoded_text_original"), ("post", "decoded_text")):
            preds = model.predict_entities(str(r[col]), LABELS, threshold=THRESHOLD)
            ents[key] = {norm(p["text"]) for p in preds if norm(p["text"])}
        raw, post = ents["raw"], ents["post"]
        new, lost, kept = post - raw, raw - post, raw & post
        rows.append({
            "node_id": r["node_id"],
            "n_ent_raw": len(raw), "n_ent_post": len(post),
            "n_kept": len(kept), "n_new": len(new), "n_lost": len(lost),
            "new_rate": round(len(new) / len(post), 4) if post else 0.0,
            "lost_rate": round(len(lost) / len(raw), 4) if raw else 0.0,
            "new_entities": " | ".join(sorted(new)),
            "lost_entities": " | ".join(sorted(lost)),
        })
        print(f"[{i+1}/{len(df)}] {r['node_id']}: raw={len(raw)} post={len(post)} "
              f"new={len(new)} lost={len(lost)}")

    out = pd.DataFrame(rows)
    path = f"{D}/data/postproc_gliner_entities.csv"
    out.to_csv(path, index=False)

    tot_raw, tot_post = out.n_ent_raw.sum(), out.n_ent_post.sum()
    tot_new, tot_lost = out.n_new.sum(), out.n_lost.sum()
    print("\n=== corpus level (unique-per-node 합계) ===")
    print(f"raw entities  = {tot_raw}")
    print(f"post entities = {tot_post}")
    print(f"new  = {tot_new}  -> micro new-rate  = {tot_new/max(1,tot_post):.4f}")
    print(f"lost = {tot_lost}  -> micro lost-rate = {tot_lost/max(1,tot_raw):.4f}")
    print(f"macro new_rate  mean={out.new_rate.mean():.4f} max={out.new_rate.max():.4f}")
    print(f"macro lost_rate mean={out.lost_rate.mean():.4f} max={out.lost_rate.max():.4f}")
    print(f"신규 개체가 1개 이상 생긴 노드 = {(out.n_new>0).sum()}/{len(out)}")
    print(f"[OK] saved {path}")


if __name__ == "__main__":
    main()
