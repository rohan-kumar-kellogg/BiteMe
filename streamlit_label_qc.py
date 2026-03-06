import os
import pandas as pd
import streamlit as st


SUSPECTS_CSV = "reports/suspected_mislabeled.csv"
MANIFEST_CSV = "images/manifest.csv"
MANIFEST_CLEAN_CSV = "images/manifest_clean.csv"


def _label_col(df: pd.DataFrame) -> str:
    for c in ["dish_label", "dish_class", "dish_family"]:
        if c in df.columns:
            return c
    raise ValueError("manifest needs one of: dish_label, dish_class, dish_family")


def _init_state():
    if "decisions" not in st.session_state:
        st.session_state.decisions = {}


def _set_decision(image_path: str, action: str, new_label: str | None = None):
    st.session_state.decisions[image_path] = {"action": action, "new_label": new_label}


def _apply_decisions(manifest: pd.DataFrame, label_col: str):
    out = manifest.copy()
    decisions = st.session_state.decisions
    keep_rows = []
    for _, r in out.iterrows():
        ip = str(r["image_path"])
        d = decisions.get(ip)
        if not d:
            keep_rows.append(r)
            continue
        if d["action"] == "remove":
            continue
        if d["action"] == "relabel" and d.get("new_label"):
            r[label_col] = str(d["new_label"])
            if "dish_label" in out.columns:
                r["dish_label"] = str(d["new_label"])
            if "dish_class" in out.columns and label_col == "dish_class":
                r["dish_class"] = str(d["new_label"])
        keep_rows.append(r)
    return pd.DataFrame(keep_rows)


def main():
    st.set_page_config(page_title="Label QC", layout="wide")
    st.title("Label QC - Suspected Mislabeled Images")
    _init_state()

    if not os.path.exists(SUSPECTS_CSV):
        st.error(f"Missing {SUSPECTS_CSV}. Run utils/data_quality_report.py first.")
        return
    if not os.path.exists(MANIFEST_CSV):
        st.error(f"Missing {MANIFEST_CSV}.")
        return

    suspects = pd.read_csv(SUSPECTS_CSV)
    manifest = pd.read_csv(MANIFEST_CSV)
    label_col = _label_col(manifest)

    st.caption(f"Loaded {len(suspects)} suspected rows. Manifest label column: `{label_col}`")
    if len(suspects) == 0:
        st.info("No suspected mislabeled rows found.")
        return

    idx = st.slider("Review index", min_value=0, max_value=len(suspects) - 1, value=0, step=1)
    row = suspects.iloc[idx]
    image_path = str(row.get("image_path", ""))
    self_label = str(row.get("self_label", ""))
    nn_labels = str(row.get("neighbor_labels_topk", ""))
    majority = str(row.get("majority_neighbor_label", ""))
    decision = st.session_state.decisions.get(image_path, {})

    c1, c2 = st.columns([2, 3])
    with c1:
        if os.path.exists(image_path):
            st.image(image_path, use_container_width=True)
        else:
            st.warning(f"Image missing: {image_path}")
    with c2:
        st.markdown(f"**Image:** `{image_path}`")
        st.markdown(f"**Current label:** `{self_label}`")
        st.markdown(f"**Majority neighbor label:** `{majority}`")
        st.markdown(f"**Top-5 neighbor labels:** `{nn_labels}`")
        st.markdown(f"**Saved decision:** `{decision.get('action', 'none')}`")

    labels = sorted(manifest[label_col].dropna().astype(str).unique().tolist())
    relabel_value = st.selectbox("Relabel dropdown", labels, index=labels.index(self_label) if self_label in labels else 0)

    b1, b2, b3 = st.columns(3)
    with b1:
        if st.button("Keep"):
            _set_decision(image_path, "keep", None)
            st.success("Saved: Keep")
    with b2:
        if st.button("Relabel"):
            _set_decision(image_path, "relabel", relabel_value)
            st.success(f"Saved: Relabel -> {relabel_value}")
    with b3:
        if st.button("Remove"):
            _set_decision(image_path, "remove", None)
            st.warning("Saved: Remove")

    st.divider()
    st.write(f"Total decisions saved: {len(st.session_state.decisions)}")
    if st.button("Write images/manifest_clean.csv"):
        clean = _apply_decisions(manifest, label_col)
        clean.to_csv(MANIFEST_CLEAN_CSV, index=False)
        st.success(f"Wrote {MANIFEST_CLEAN_CSV} with {len(clean)} rows (from {len(manifest)}).")
        st.info(
            "Next steps: replace manifest.csv with manifest_clean.csv (or point pipeline to it), "
            "run utils/data_generator.py, then run utils/eval_retrieval.py."
        )


if __name__ == "__main__":
    main()

