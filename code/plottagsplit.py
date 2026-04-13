"""
BERTopic Hierarchy Extractor
----------------------------
Loads saved BERTopic models and exports their hierarchical topic structure
to a JSON file that can be loaded by the interactive visualizer.

Works even for models saved WITHOUT calling model.hierarchical_topics(docs)
first — it recomputes the hierarchy from the stored c-TF-IDF matrix.

Usage:
    python extract_bertopic_hierarchy.py
"""

import json
import os
import numpy as np
import pandas as pd
from bertopic import BERTopic
from scipy.cluster.hierarchy import linkage, to_tree
from scipy.spatial.distance import pdist

# ── CONFIG ────────────────────────────────────────────────────────────────────
MODEL_PATHS = {
    "Couple":                     "results/Couple_bertopic_model",
    "Family with young children": "results/Family_with_young_children_bertopic_model",
    "Group":                      "results/Group_bertopic_model",
    "Solo traveler":              "results/Solo_traveler_bertopic_model",
}

CSV_PATHS = {
    "Couple":                     "results/Couple_topics.csv",
    "Family with young children": "results/Family_with_young_children_topics.csv",
    "Group":                      "results/Group_topics.csv",
    "Solo traveler":              "results/Solo_traveler_topics.csv",
}

OUTPUT_FILE = "bertopic_hierarchy.json"
# ── END CONFIG ────────────────────────────────────────────────────────────────


def safe_float(val):
    try:
        f = float(val)
        return None if (np.isnan(f) or np.isinf(f)) else round(f, 4)
    except Exception:
        return None


# ── Topic metadata ────────────────────────────────────────────────────────────

def build_topic_meta(model, csv_path=None):
    topic_info = model.get_topic_info()
    topic_info = topic_info[topic_info["Topic"] != -1].copy()

    sent_map = {}
    if csv_path and os.path.exists(csv_path):
        try:
            csv_df = pd.read_csv(csv_path)
            csv_df = csv_df[csv_df["Topic"] != -1]
            for _, row in csv_df.iterrows():
                tid = int(row["Topic"])
                sent_map[tid] = {
                    "sentiment":     str(row.get("Sentiment", "")),
                    "net_sentiment": safe_float(row.get("Net_Sentiment")),
                }
        except Exception as e:
            print(f"     ⚠ Could not read CSV {csv_path}: {e}")

    topic_meta = {}
    for _, row in topic_info.iterrows():
        tid   = int(row["Topic"])
        label = str(row.get("Name", f"Topic {tid}"))
        count = int(row.get("Count", 0))
        words = model.get_topic(tid)
        top_words = [w for w, _ in words[:8]] if words else []
        entry = {"id": tid, "label": label, "count": count, "top_words": top_words}
        if tid in sent_map:
            entry.update(sent_map[tid])
        topic_meta[tid] = entry

    return topic_meta


# ── Hierarchy from c-TF-IDF ───────────────────────────────────────────────────

def compute_hierarchy_from_ctfidf(model, topic_meta):
    """
    Reconstruct a dendrogram using the stored c-TF-IDF topic vectors.
    This is the same approach BERTopic uses internally — we just do it
    post-hoc from the saved matrix (save_ctfidf=True must have been used).
    """
    ctfidf = model.c_tf_idf_
    topic_ids_ordered = sorted(topic_meta.keys())   # [0, 1, 2, ...]

    # Row 0 in c_tf_idf_ is the outlier topic (-1); real topics start at row 1
    try:
        matrix = np.asarray(ctfidf[
            [tid + 1 for tid in topic_ids_ordered], :
        ].todense())
    except Exception:
        matrix = np.array(ctfidf[[tid + 1 for tid in topic_ids_ordered], :])

    # L2-normalise so cosine distance is meaningful
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms[norms == 0] = 1
    matrix /= norms

    dists = np.clip(pdist(matrix, metric="cosine"), 0, None)
    Z     = linkage(dists, method="ward")
    root, _ = to_tree(Z, rd=True)
    return root, topic_ids_ordered


def scipy_node_to_dict(node, topic_ids_ordered, topic_meta, label_override=None):
    n = len(topic_ids_ordered)

    if node.is_leaf():
        idx = node.id
        if idx >= n:
            return None
        tid = topic_ids_ordered[idx]
        m   = topic_meta[tid]
        return {
            "name":          m["label"],
            "topic_id":      m["id"],
            "count":         m["count"],
            "top_words":     m["top_words"],
            "sentiment":     m.get("sentiment", ""),
            "net_sentiment": m.get("net_sentiment"),
            "children":      [],
        }

    left  = scipy_node_to_dict(node.left,  topic_ids_ordered, topic_meta)
    right = scipy_node_to_dict(node.right, topic_ids_ordered, topic_meta)
    children = [c for c in [left, right] if c is not None]

    total_count  = sum(c.get("count", 0) for c in children)
    all_words    = []
    for c in children:
        all_words.extend(c.get("top_words", []))
    unique_words = list(dict.fromkeys(all_words))[:8]
    name = label_override or ("_".join(unique_words[:3]) if unique_words else "cluster")

    return {
        "name":      name,
        "topic_id":  None,
        "count":     total_count,
        "top_words": unique_words,
        "children":  children,
    }


# ── Tree builders ─────────────────────────────────────────────────────────────

def build_tree(model, topic_meta, traveler_label):
    # 1. Stored hierarchy (models re-saved after fix)
    hier = getattr(model, "hierarchical_topics_", None)
    if hier is not None and not (hasattr(hier, "empty") and hier.empty):
        print("   ✓ Using stored hierarchical_topics_")
        return _from_stored_hier(model, topic_meta, traveler_label)

    # 2. Recompute from c-TF-IDF (works for your existing models)
    if getattr(model, "c_tf_idf_", None) is not None and len(topic_meta) >= 2:
        print("   ↻ Recomputing hierarchy from c-TF-IDF...")
        try:
            root_node, tids = compute_hierarchy_from_ctfidf(model, topic_meta)
            tree = scipy_node_to_dict(root_node, tids, topic_meta,
                                      label_override=traveler_label)
            tree["name"] = traveler_label
            print("   ✓ Done")
            return tree
        except Exception as e:
            print(f"   ⚠ Failed ({e}), using flat layout")

    # 3. Last resort
    print("   ⚠ Flat layout (no c-TF-IDF available)")
    children = sorted(
        [{"name": m["label"], "topic_id": m["id"], "count": m["count"],
          "top_words": m["top_words"], "sentiment": m.get("sentiment",""),
          "net_sentiment": m.get("net_sentiment"), "children": []}
         for m in topic_meta.values()],
        key=lambda x: -x["count"],
    )
    return {"name": traveler_label, "topic_id": None,
            "count": sum(m["count"] for m in topic_meta.values()),
            "top_words": [], "children": children, "flat": True}


def _from_stored_hier(model, topic_meta, traveler_label):
    hier = model.hierarchical_topics_
    parent_to_children, all_child_ids = {}, set()
    for _, row in hier.iterrows():
        pid = int(row["Parent_ID"])
        if pid not in parent_to_children:
            parent_to_children[pid] = {"name": str(row["Parent_Name"]), "children": []}
        for cid in [int(row["Child_Left_ID"]), int(row["Child_Right_ID"])]:
            all_child_ids.add(cid)
            parent_to_children[pid]["children"].append(cid)
    root_id = max(set(parent_to_children) - all_child_ids)

    def node(nid):
        if nid in topic_meta:
            m = topic_meta[nid]
            return {"name": m["label"], "topic_id": m["id"], "count": m["count"],
                    "top_words": m["top_words"], "sentiment": m.get("sentiment",""),
                    "net_sentiment": m.get("net_sentiment"), "children": []}
        if nid in parent_to_children:
            info = parent_to_children[nid]
            kids = [node(c) for c in info["children"]]
            words = list(dict.fromkeys(w for c in kids for w in c.get("top_words",[])))[:8]
            return {"name": info["name"], "topic_id": nid,
                    "count": sum(c.get("count",0) for c in kids),
                    "top_words": words, "children": kids}
        return {"name": f"Node {nid}", "topic_id": nid, "count": 0, "top_words": [], "children": []}

    tree = node(root_id)
    tree["name"] = traveler_label
    return tree


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    output = {"models": []}
    for label, path in MODEL_PATHS.items():
        if not os.path.exists(path):
            print(f"⚠  Not found, skipping: {path}")
            continue
        print(f"\nLoading: {label}")
        try:
            model    = BERTopic.load(path)
            csv_path = CSV_PATHS.get(label)
            meta     = build_topic_meta(model, csv_path)
            tree     = build_tree(model, meta, label)
            output["models"].append(tree)
            print(f"   ✓ {len(meta)} topics")
        except Exception as e:
            print(f"   ✗ Failed: {e}")

    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)
    print(f"\n✅ Saved → {OUTPUT_FILE}")

if __name__ == "__main__":
    main()