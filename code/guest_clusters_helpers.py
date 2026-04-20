from __future__ import annotations

import ast
import itertools
import math
import re
from pathlib import Path
import sys
from typing import Iterable, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment
from scipy.stats import kruskal, mannwhitneyu
from sklearn.cluster import KMeans
from sklearn.compose import ColumnTransformer
from sklearn.metrics import adjusted_rand_score, calinski_harabasz_score, davies_bouldin_score, silhouette_score
from sklearn.preprocessing import OneHotEncoder

CURRENT_DIR = Path(__file__).resolve().parent
LIB_DIR = CURRENT_DIR / "lib"
if str(LIB_DIR) not in sys.path:
    sys.path.insert(0, str(LIB_DIR))

from rp_paths import (
    find_project_root,
    resolve_hotel_document_info_csv,
    resolve_output_dir,
    resolve_stratified_reviews_csv,
)

ZERO_SHOT_MAJOR_TOPICS = [
    "Staff Service",
    "Room Comfort & Quality",
    "Cleanliness",
    "Location & Accessibility",
    "Breakfast & Food",
    "Bathroom & Shower Experience",
    "Noise & Sleep Disturbance",
    "Facilities & Amenities",
    "Value for Money",
    "Maintenance & Room Condition",
]

TAG_REVIEW_COLUMNS = [
    "ID",
    "Hotel_Name",
    "City",
    "Review_Date",
    "Reviewer_Score",
    "Average_Score",
    "Tags",
]


def find_repo_root(start: Optional[Path] = None) -> Path:
    return find_project_root(start)


def output_dir_for_repo(repo_root: Path) -> Path:
    return resolve_output_dir(
        "RP_GUEST_CLUSTER_OUTPUT_DIR",
        "outputs/intermediate/guest_clusters",
    )


def assign_season(month: object) -> object:
    if pd.isna(month):
        return pd.NA
    month = int(month)
    if month in (12, 1, 2):
        return "Winter"
    if month in (3, 4, 5):
        return "Spring"
    if month in (6, 7, 8):
        return "Summer"
    return "Autumn"


def load_review_data(repo_root: Path, valid_person_ids: Optional[pd.Series] = None) -> pd.DataFrame:
    review_path = resolve_stratified_reviews_csv()
    df = pd.read_csv(review_path, usecols=TAG_REVIEW_COLUMNS)
    df["Review_Date"] = pd.to_datetime(df["Review_Date"], dayfirst=False, errors="coerce")
    df["Season"] = df["Review_Date"].dt.month.map(assign_season)
    df["Person_id"] = pd.to_numeric(df["ID"], errors="coerce").astype("Int64")
    df = df.dropna(subset=["Person_id"]).drop_duplicates(subset=["Person_id"]).copy()
    if valid_person_ids is not None:
        df = df[df["Person_id"].isin(pd.Index(valid_person_ids.dropna().astype("Int64").unique()))].copy()
    return df


def load_doc_info(repo_root: Path) -> pd.DataFrame:
    doc_path = resolve_hotel_document_info_csv()
    doc = pd.read_csv(doc_path, usecols=["Person_id", "Semantic_Label"])
    doc["Person_id"] = pd.to_numeric(doc["Person_id"], errors="coerce").astype("Int64")
    doc = doc.dropna(subset=["Person_id"]).copy()
    return doc[doc["Semantic_Label"].isin(ZERO_SHOT_MAJOR_TOPICS)].copy()


def parse_tags(tags_str: object) -> list[str]:
    if not isinstance(tags_str, str) or not tags_str.strip():
        return []
    try:
        parsed = ast.literal_eval(tags_str)
    except (SyntaxError, ValueError):
        parsed = tags_str.split(",")
    return [" ".join(str(tag).strip().split()) for tag in parsed if str(tag).strip()]


def normalize_tag(tag: str) -> str:
    return re.sub(r"\s+", " ", tag.strip().lower())


def keep_semantic_tag(tag: str) -> bool:
    lowered = normalize_tag(tag)
    if lowered == "submitted from a mobile device":
        return False
    if re.fullmatch(r"stayed\s+\d+\s+nights?", lowered):
        return False
    if re.fullmatch(r"\d+\s+rooms?", lowered):
        return False
    return True


def extract_trip_purpose(tags_lower: Sequence[str]) -> str:
    leisure = any("leisure trip" in tag for tag in tags_lower)
    business = any("business trip" in tag for tag in tags_lower)
    if leisure and business:
        return "mixed"
    if leisure:
        return "leisure"
    if business:
        return "business"
    return "other"


def extract_party_type(tags_lower: Sequence[str]) -> str:
    if any("family with young children" in tag for tag in tags_lower):
        return "family_young_children"
    if any("family with older children" in tag for tag in tags_lower):
        return "family_older_children"
    if any("solo traveler" in tag or "solo traveller" in tag or "single" in tag for tag in tags_lower):
        return "solo"
    if any("couple" in tag for tag in tags_lower):
        return "couple"
    if any("group" in tag or "friends" in tag for tag in tags_lower):
        return "group_friends"
    return "other"


def extract_stay_nights(tags_lower: Sequence[str]) -> Optional[int]:
    for tag in tags_lower:
        match = re.search(r"stayed\s+(\d+)\s+night", tag)
        if match:
            return int(match.group(1))
    return None


def bucket_stay_nights(value: Optional[int]) -> str:
    if value is None:
        return "unknown"
    if value <= 1:
        return "1"
    if value == 2:
        return "2"
    if value == 3:
        return "3"
    if 4 <= value <= 5:
        return "4_5"
    return "6_plus"


def extract_room_count(tags_lower: Sequence[str]) -> int:
    for tag in tags_lower:
        match = re.fullmatch(r"(\d+)\s+rooms?", tag)
        if match:
            return int(match.group(1))
    return 1


def bucket_room_count(value: int) -> str:
    if value <= 1:
        return "1"
    if value == 2:
        return "2"
    return "3_plus"


def extract_room_type(tags_original: Sequence[str], tags_lower: Sequence[str]) -> str:
    room_tags = []
    for original, lowered in zip(tags_original, tags_lower):
        if lowered in {"leisure trip", "business trip", "submitted from a mobile device"}:
            continue
        if lowered.startswith("stayed "):
            continue
        if re.fullmatch(r"\d+\s+rooms?", lowered):
            continue
        if (
            "solo traveler" in lowered
            or "solo traveller" in lowered
            or "single" in lowered
            or "couple" in lowered
            or "group" in lowered
            or "friends" in lowered
            or "family with young children" in lowered
            or "family with older children" in lowered
        ):
            continue
        room_tags.append(original)
    return room_tags[0] if room_tags else "Unknown room"


def parse_reviewer_metadata(review_df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for row in review_df.itertuples(index=False):
        tags_original = parse_tags(row.Tags)
        tags_lower = [normalize_tag(tag) for tag in tags_original]
        stay_nights = extract_stay_nights(tags_lower)
        rows.append(
            {
                "Person_id": row.Person_id,
                "Hotel_Name": row.Hotel_Name,
                "City": row.City,
                "Season": row.Season,
                "Review_Date": row.Review_Date,
                "Reviewer_Score": row.Reviewer_Score,
                "Average_Score": row.Average_Score,
                "Tags": row.Tags,
                "Parsed_Tags": tags_original,
                "Trip_Purpose": extract_trip_purpose(tags_lower),
                "Party_Type": extract_party_type(tags_lower),
                "Stay_Nights": stay_nights,
                "Stay_Bucket": bucket_stay_nights(stay_nights),
                "Room_Count_Bucket": bucket_room_count(extract_room_count(tags_lower)),
                "Submitted_Mobile": any("submitted from a mobile device" in tag for tag in tags_lower),
                "Room_Type_Raw": extract_room_type(tags_original, tags_lower),
            }
        )
    out = pd.DataFrame(rows)
    out["Person_id"] = out["Person_id"].astype("Int64")
    return out


def collapse_sparse_levels(series: pd.Series, min_share: float = 0.02) -> pd.Series:
    min_count = max(1, math.ceil(len(series) * min_share))
    counts = series.value_counts(dropna=False)
    keep = counts[counts >= min_count].index
    return series.where(series.isin(keep), other="other")


def prepare_cluster_features(metadata: pd.DataFrame, top_n_room_types: int = 10) -> pd.DataFrame:
    room_type_top = metadata["Room_Type_Raw"].value_counts().head(top_n_room_types).index
    out = metadata[
        ["Person_id", "Trip_Purpose", "Party_Type", "Stay_Bucket", "Room_Count_Bucket", "Submitted_Mobile", "Room_Type_Raw"]
    ].copy()
    out["Trip_Purpose"] = collapse_sparse_levels(out["Trip_Purpose"])
    out["Party_Type"] = collapse_sparse_levels(out["Party_Type"])
    out["Room_Type"] = out["Room_Type_Raw"].where(out["Room_Type_Raw"].isin(room_type_top), other="other_room")
    out["Submitted_Mobile"] = out["Submitted_Mobile"].map({True: "mobile", False: "not_mobile"})
    return out[["Person_id", "Trip_Purpose", "Party_Type", "Stay_Bucket", "Room_Count_Bucket", "Submitted_Mobile", "Room_Type"]]


def build_feature_matrix(feature_df: pd.DataFrame) -> tuple[ColumnTransformer, np.ndarray, list[str]]:
    cols = ["Trip_Purpose", "Party_Type", "Stay_Bucket", "Room_Count_Bucket", "Submitted_Mobile", "Room_Type"]
    encoder = ColumnTransformer(
        [("categorical", OneHotEncoder(handle_unknown="ignore", sparse_output=False), cols)],
        sparse_threshold=0.0,
    )
    matrix = encoder.fit_transform(feature_df[cols])
    names = list(encoder.get_feature_names_out())
    return encoder, np.asarray(matrix, dtype=float), names


def _row_normalize(matrix: np.ndarray) -> np.ndarray:
    matrix = np.asarray(matrix, dtype=float)
    if matrix.ndim != 2:
        raise ValueError("Expected a 2D matrix to normalize.")
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms = np.where(norms == 0.0, 1.0, norms)
    return matrix / norms


def build_semantic_tag_matrix(
    metadata: pd.DataFrame,
    tags_column: str = "Parsed_Tags",
    model_name: str = "all-MiniLM-L6-v2",
    device: Optional[str] = None,
    batch_size: int = 128,
    min_tag_df: int = 1,
    use_idf_weighting: bool = True,
    filter_nonsemantic_tags: bool = True,
    show_progress_bar: bool = False,
) -> tuple[np.ndarray, pd.DataFrame]:
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError as exc:
        raise ImportError("sentence-transformers is required to build semantic tag features.") from exc

    normalized_tag_lists: list[list[str]] = []
    tag_presence_rows: list[dict[str, object]] = []
    for row_idx, value in enumerate(metadata[tags_column].tolist()):
        if isinstance(value, str):
            raw_tags = parse_tags(value)
        elif isinstance(value, Sequence):
            raw_tags = [" ".join(str(tag).strip().split()) for tag in value if str(tag).strip()]
        else:
            raw_tags = []

        normalized_tags = []
        seen = set()
        for raw_tag in raw_tags:
            normalized = normalize_tag(raw_tag)
            if filter_nonsemantic_tags and not keep_semantic_tag(normalized):
                continue
            normalized_tags.append(normalized)
            if normalized not in seen:
                tag_presence_rows.append(
                    {
                        "row_index": row_idx,
                        "Normalized_Tag": normalized,
                        "Display_Tag": raw_tag,
                    }
                )
                seen.add(normalized)
        normalized_tag_lists.append(normalized_tags)

    if not tag_presence_rows:
        empty = pd.DataFrame(columns=["Normalized_Tag", "Display_Tag", "Document_Frequency", "IDF_Weight"])
        return np.zeros((len(metadata), 0), dtype=float), empty

    tag_presence = pd.DataFrame(tag_presence_rows)
    tag_frequency = tag_presence["Normalized_Tag"].value_counts().sort_index()
    if min_tag_df > 1:
        keep = tag_frequency[tag_frequency >= min_tag_df].index
        tag_presence = tag_presence[tag_presence["Normalized_Tag"].isin(keep)].copy()
        normalized_tag_lists = [[tag for tag in tags if tag in set(keep)] for tags in normalized_tag_lists]
        tag_frequency = tag_presence["Normalized_Tag"].value_counts().sort_index()

    if tag_presence.empty:
        empty = pd.DataFrame(columns=["Normalized_Tag", "Display_Tag", "Document_Frequency", "IDF_Weight"])
        return np.zeros((len(metadata), 0), dtype=float), empty

    tag_display = (
        tag_presence.groupby("Normalized_Tag")["Display_Tag"].agg(lambda series: series.value_counts().index[0]).sort_index()
    )
    idf_weights = pd.Series(
        np.log((1 + len(metadata)) / (1 + tag_frequency.astype(float))) + 1.0,
        index=tag_frequency.index,
        dtype=float,
    )

    try:
        model = SentenceTransformer(model_name, device=device)
        tag_vocab = tag_frequency.index.tolist()
        embeddings = model.encode(
            tag_vocab,
            batch_size=batch_size,
            convert_to_numpy=True,
            show_progress_bar=show_progress_bar,
        )
    except Exception as exc:
        raise RuntimeError(
            f"Could not load or encode semantic tags with model '{model_name}'. "
            "Ensure the model is installed or already cached locally."
        ) from exc

    embeddings = _row_normalize(np.asarray(embeddings, dtype=float))
    embedding_lookup = {tag: embeddings[idx] for idx, tag in enumerate(tag_vocab)}
    weight_lookup = {tag: float(idf_weights.loc[tag]) if use_idf_weighting else 1.0 for tag in tag_vocab}

    semantic_matrix = np.zeros((len(metadata), embeddings.shape[1]), dtype=float)
    for row_idx, tags in enumerate(normalized_tag_lists):
        ordered_unique_tags = [tag for tag in dict.fromkeys(tags) if tag in embedding_lookup]
        if not ordered_unique_tags:
            continue
        row_embeddings = np.vstack([embedding_lookup[tag] for tag in ordered_unique_tags])
        row_weights = np.asarray([weight_lookup[tag] for tag in ordered_unique_tags], dtype=float)
        semantic_matrix[row_idx] = np.average(row_embeddings, axis=0, weights=row_weights)

    semantic_matrix = _row_normalize(semantic_matrix)
    tag_summary = pd.DataFrame(
        {
            "Normalized_Tag": tag_frequency.index,
            "Display_Tag": tag_display.reindex(tag_frequency.index).to_numpy(),
            "Document_Frequency": tag_frequency.to_numpy(),
            "IDF_Weight": idf_weights.reindex(tag_frequency.index).to_numpy(),
        }
    )
    return semantic_matrix, tag_summary


def build_hybrid_feature_matrix(
    feature_df: pd.DataFrame,
    semantic_matrix: np.ndarray,
    structured_weight: float = 1.0,
    semantic_weight: float = 1.0,
    normalize_blocks: bool = True,
) -> tuple[ColumnTransformer, np.ndarray, list[str]]:
    encoder, structured_matrix, structured_names = build_feature_matrix(feature_df)
    semantic_matrix = np.asarray(semantic_matrix, dtype=float)
    if semantic_matrix.ndim != 2:
        raise ValueError("semantic_matrix must be a 2D array.")
    if semantic_matrix.shape[0] != structured_matrix.shape[0]:
        raise ValueError("semantic_matrix must have the same number of rows as feature_df.")

    if normalize_blocks:
        structured_block = _row_normalize(structured_matrix)
        semantic_block = _row_normalize(semantic_matrix) if semantic_matrix.shape[1] else semantic_matrix
    else:
        structured_block = structured_matrix
        semantic_block = semantic_matrix

    blocks = [structured_weight * structured_block]
    feature_names = list(structured_names)
    if semantic_block.shape[1]:
        blocks.append(semantic_weight * semantic_block)
        feature_names.extend([f"semantic_tag_dim_{idx + 1}" for idx in range(semantic_block.shape[1])])
    combined = np.hstack(blocks)
    return encoder, combined, feature_names


def _silhouette_stats(matrix: np.ndarray, labels: np.ndarray, repeats: int = 5, sample_size: int = 5000, random_state: int = 42) -> Tuple[float, float]:
    rng = np.random.default_rng(random_state)
    scores = []
    actual_size = min(sample_size, len(labels))
    for _ in range(repeats):
        idx = np.arange(len(labels)) if actual_size == len(labels) else rng.choice(len(labels), size=actual_size, replace=False)
        if np.unique(labels[idx]).size < 2:
            continue
        scores.append(silhouette_score(matrix[idx], labels[idx], metric="euclidean"))
    if not scores:
        return float("nan"), float("nan")
    return float(np.mean(scores)), float(np.std(scores, ddof=1) if len(scores) > 1 else 0.0)


def select_cluster_solution(matrix: np.ndarray, k_values: Iterable[int] = range(2, 9), min_cluster_share: float = 0.05, random_state: int = 42) -> pd.DataFrame:
    rows = []
    for k in k_values:
        model = KMeans(n_clusters=k, n_init=20, random_state=random_state)
        labels = model.fit_predict(matrix)
        counts = pd.Series(labels).value_counts().sort_index()
        sil_mean, sil_sd = _silhouette_stats(matrix, labels, random_state=random_state + k)
        rows.append(
            {
                "k": k,
                "silhouette_mean": sil_mean,
                "silhouette_sd": sil_sd,
                "calinski_harabasz": calinski_harabasz_score(matrix, labels),
                "davies_bouldin": davies_bouldin_score(matrix, labels),
                "min_cluster_size": int(counts.min()),
                "min_cluster_share": float(counts.min() / len(labels)),
                "labels": labels,
                "model": model,
            }
        )
    metrics = pd.DataFrame(rows)
    valid = metrics[metrics["min_cluster_share"] >= min_cluster_share].copy()
    if valid.empty:
        raise ValueError("No cluster solution met the minimum cluster share threshold.")
    best = valid.loc[valid["silhouette_mean"].idxmax()]
    cutoff = best["silhouette_mean"] - best["silhouette_sd"]
    chosen = valid[valid["silhouette_mean"] >= cutoff].sort_values("k").iloc[0]
    metrics["selected"] = metrics["k"].eq(chosen["k"])
    return metrics


def reorder_clusters(labels: np.ndarray, centers: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    size_order = pd.Series(labels).value_counts().sort_values(ascending=False).index.tolist()
    mapping = {raw: idx for idx, raw in enumerate(size_order)}
    ordered_labels = np.array([mapping[label] for label in labels], dtype=int)
    ordered_centers = np.vstack([centers[raw] for raw in size_order])
    return ordered_labels, ordered_centers


def build_cluster_assignments(metadata: pd.DataFrame, labels: np.ndarray) -> pd.DataFrame:
    out = metadata.copy()
    out["Cluster_ID"] = labels + 1
    out["Cluster"] = out["Cluster_ID"].map(lambda value: f"Cluster {value}")
    return out


def _format_profile_value(column: str, value: object) -> str:
    mapping = {
        "Trip_Purpose": {"leisure": "Leisure", "business": "Business", "mixed": "Mixed", "other": "Other"},
        "Party_Type": {
            "family_young_children": "Young Families",
            "family_older_children": "Older Families",
            "solo": "Solo Travellers",
            "couple": "Couples",
            "group_friends": "Groups",
            "other": "Other Guests",
        },
        "Stay_Bucket": {
            "1": "1-Night",
            "2": "2-Night",
            "3": "3-Night",
            "4_5": "4-5 Night",
            "6_plus": "Long-Stay",
            "unknown": "Unknown Stay",
        },
        "Room_Count_Bucket": {
            "1": "1 Room",
            "2": "2 Rooms",
            "3_plus": "3+ Rooms",
        },
    }
    text = str(value)
    return mapping.get(column, {}).get(text, text.replace("_", " ").title())


def _top_lifted_value(
    subset: pd.Series,
    overall: pd.Series,
    column: str,
    excluded: Optional[set[str]] = None,
    min_share: float = 0.15,
) -> str:
    distribution = subset.value_counts(normalize=True)
    candidates = []
    for value, share in distribution.items():
        if pd.isna(value):
            continue
        text = str(value)
        if excluded and text in excluded:
            continue
        global_share = float(overall.get(value, 0.0))
        delta = share - global_share
        candidates.append((delta, share, _format_profile_value(column, value)))
    if not candidates:
        return ""
    candidates.sort(key=lambda row: (row[0], row[1]), reverse=True)
    positive = [row for row in candidates if row[1] >= min_share and row[0] > 0]
    return positive[0][2] if positive else candidates[0][2]


def _prepare_tag_profiles(assignments: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, str], pd.Series]:
    tag_frame = assignments[["Person_id", "Cluster_ID", "Parsed_Tags"]].explode("Parsed_Tags").dropna().copy()
    if tag_frame.empty:
        return tag_frame, {}, pd.Series(dtype=float)
    tag_frame["Normalized_Tag"] = tag_frame["Parsed_Tags"].map(normalize_tag)
    tag_frame = tag_frame[tag_frame["Normalized_Tag"].map(keep_semantic_tag)].copy()
    if tag_frame.empty:
        return tag_frame, {}, pd.Series(dtype=float)
    tag_frame = tag_frame.drop_duplicates(subset=["Person_id", "Cluster_ID", "Normalized_Tag"])
    tag_display = tag_frame.groupby("Normalized_Tag")["Parsed_Tags"].agg(lambda series: series.value_counts().index[0]).to_dict()
    overall_share = tag_frame.groupby("Normalized_Tag")["Person_id"].nunique() / assignments["Person_id"].nunique()
    return tag_frame, tag_display, overall_share


def _distinctive_tags_for_cluster(
    tag_frame: pd.DataFrame,
    tag_display: Mapping[str, str],
    overall_share: pd.Series,
    cluster_id: int,
    cluster_size: int,
    top_n: int = 3,
) -> str:
    if tag_frame.empty or cluster_size <= 0:
        return ""
    cluster_rows = tag_frame[tag_frame["Cluster_ID"] == cluster_id]
    if cluster_rows.empty:
        return ""
    cluster_share = cluster_rows.groupby("Normalized_Tag")["Person_id"].nunique() / cluster_size
    rows = []
    for tag, share in cluster_share.items():
        global_share = float(overall_share.get(tag, 0.0))
        rows.append((share - global_share, share, tag_display.get(tag, tag)))
    rows.sort(key=lambda row: (row[0], row[1]), reverse=True)
    positive = [display for delta, _, display in rows if delta > 0]
    selected = positive[:top_n] if positive else [display for _, _, display in rows[:top_n]]
    return ", ".join(selected)


def _resolve_cluster_name_duplicates(summary: pd.DataFrame) -> pd.DataFrame:
    seen = set()
    names = []
    for row in summary.itertuples(index=False):
        proposed = row.Cluster_Name
        if proposed not in seen:
            names.append(proposed)
            seen.add(proposed)
            continue
        room_suffix = row.Primary_Room_Type if row.Primary_Room_Type != "Unknown room" else ""
        candidate = f"{proposed} {room_suffix}".strip() if room_suffix else proposed
        if candidate in seen:
            candidate = f"{proposed} C{row.Cluster_ID}"
        names.append(candidate)
        seen.add(candidate)
    out = summary.copy()
    out["Cluster_Name"] = names
    return out


def summarize_clusters(assignments: pd.DataFrame) -> pd.DataFrame:
    total = len(assignments)
    overall_trip = assignments["Trip_Purpose"].value_counts(normalize=True)
    overall_party = assignments["Party_Type"].value_counts(normalize=True)
    overall_stay = assignments["Stay_Bucket"].value_counts(normalize=True)
    overall_room_count = assignments["Room_Count_Bucket"].value_counts(normalize=True)
    tag_frame, tag_display, overall_tag_share = _prepare_tag_profiles(assignments)
    rows = []
    for cluster_id in sorted(assignments["Cluster_ID"].unique()):
        subset = assignments[assignments["Cluster_ID"] == cluster_id]
        primary_room_type = subset["Room_Type_Raw"].value_counts().index[0] if len(subset) else "Unknown room"
        cluster_name = " ".join(
            part
            for part in [
                _top_lifted_value(subset["Trip_Purpose"], overall_trip, "Trip_Purpose", excluded={"other"}),
                _top_lifted_value(subset["Party_Type"], overall_party, "Party_Type", excluded={"other"}),
                _top_lifted_value(subset["Stay_Bucket"], overall_stay, "Stay_Bucket", excluded={"unknown"}),
            ]
            if part
        ).strip()
        distinctive_tags = _distinctive_tags_for_cluster(tag_frame, tag_display, overall_tag_share, cluster_id, len(subset))
        rows.append(
            {
                "Cluster_ID": cluster_id,
                "Cluster": f"Cluster {cluster_id}",
                "Cluster_Name": cluster_name or f"Guest Segment {cluster_id}",
                "Review_Count": int(len(subset)),
                "Share_of_Reviews": len(subset) / total,
                "Primary_Trip_Purpose": _top_lifted_value(subset["Trip_Purpose"], overall_trip, "Trip_Purpose", excluded={"other"}),
                "Primary_Party_Type": _top_lifted_value(subset["Party_Type"], overall_party, "Party_Type", excluded={"other"}),
                "Primary_Stay_Bucket": _top_lifted_value(subset["Stay_Bucket"], overall_stay, "Stay_Bucket", excluded={"unknown"}),
                "Primary_Room_Count": _top_lifted_value(
                    subset["Room_Count_Bucket"],
                    overall_room_count,
                    "Room_Count_Bucket",
                    excluded=set(),
                    min_share=0.05,
                ),
                "Primary_Room_Type": primary_room_type,
                "Distinctive_Tags": distinctive_tags,
                "Trip_Purpose_Top": ", ".join(f"{idx} ({val:.0%})" for idx, val in subset["Trip_Purpose"].value_counts(normalize=True).head(2).items()),
                "Party_Type_Top": ", ".join(f"{idx} ({val:.0%})" for idx, val in subset["Party_Type"].value_counts(normalize=True).head(2).items()),
                "Stay_Bucket_Top": ", ".join(f"{idx} ({val:.0%})" for idx, val in subset["Stay_Bucket"].value_counts(normalize=True).head(3).items()),
                "Room_Count_Top": ", ".join(f"{idx} ({val:.0%})" for idx, val in subset["Room_Count_Bucket"].value_counts(normalize=True).head(2).items()),
                "Room_Type_Top": ", ".join(f"{idx} ({val:.0%})" for idx, val in subset["Room_Type_Raw"].value_counts(normalize=True).head(3).items()),
                "Mobile_Submission_Share": float(subset["Submitted_Mobile"].mean()),
            }
        )
    summary = _resolve_cluster_name_duplicates(pd.DataFrame(rows))
    summary["Profile_Summary"] = summary.apply(
        lambda row: (
            f"{row['Cluster']} ({row['Cluster_Name']}) covers {row['Share_of_Reviews']:.1%} of topic-tagged reviews, "
            f"dominated by {row['Party_Type_Top']} and {row['Trip_Purpose_Top']}"
            + (f", with distinctive tags such as {row['Distinctive_Tags']}." if row["Distinctive_Tags"] else ".")
        ),
        axis=1,
    )
    return summary


def attach_cluster_names(assignments: pd.DataFrame, summary_df: pd.DataFrame) -> pd.DataFrame:
    name_lookup = summary_df.set_index("Cluster_ID")["Cluster_Name"]
    out = assignments.copy()
    out["Cluster_Name"] = out["Cluster_ID"].map(name_lookup)
    out["Cluster_Label"] = out.apply(
        lambda row: f"{row['Cluster']}: {row['Cluster_Name']}" if pd.notna(row["Cluster_Name"]) else row["Cluster"],
        axis=1,
    )
    return out


def cluster_summary_markdown(summary_df: pd.DataFrame) -> str:
    lines = ["# Guest Cluster Profiles", ""]
    for row in summary_df.itertuples(index=False):
        heading = f"{row.Cluster}: {row.Cluster_Name}" if getattr(row, "Cluster_Name", "") else row.Cluster
        lines.extend(
            [
                f"## {heading}",
                f"- Reviews: {row.Review_Count:,} ({row.Share_of_Reviews:.1%})",
                f"- Generated name: {row.Cluster_Name}" if getattr(row, "Cluster_Name", "") else f"- Cluster: {row.Cluster}",
                f"- Trip purpose: {row.Trip_Purpose_Top}",
                f"- Party type: {row.Party_Type_Top}",
                f"- Stay length: {row.Stay_Bucket_Top}",
                f"- Room count: {row.Room_Count_Top}",
                f"- Room type: {row.Room_Type_Top}",
                f"- Mobile submissions: {row.Mobile_Submission_Share:.1%}",
                f"- Distinctive tags: {row.Distinctive_Tags}" if getattr(row, "Distinctive_Tags", "") else "- Distinctive tags: None",
                f"- Summary: {row.Profile_Summary}",
                "",
            ]
        )
    return "\n".join(lines).strip() + "\n"


def _relabel_bootstrap(reference_centers: np.ndarray, bootstrap_centers: np.ndarray, bootstrap_labels: np.ndarray) -> np.ndarray:
    cost = np.linalg.norm(reference_centers[:, None, :] - bootstrap_centers[None, :, :], axis=2)
    ref_idx, boot_idx = linear_sum_assignment(cost)
    mapping = {boot: ref for ref, boot in zip(ref_idx, boot_idx)}
    return np.array([mapping[label] for label in bootstrap_labels], dtype=int)


def bootstrap_cluster_stability(matrix: np.ndarray, reference_labels: np.ndarray, reference_centers: np.ndarray, n_clusters: int, n_bootstraps: int = 30, random_state: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(random_state)
    rows = []
    for iteration in range(1, n_bootstraps + 1):
        sampled_idx = rng.integers(0, matrix.shape[0], size=matrix.shape[0])
        unique_idx = np.unique(sampled_idx)
        boot_matrix = matrix[unique_idx]
        model = KMeans(n_clusters=n_clusters, n_init=10, random_state=random_state + iteration)
        boot_labels = model.fit_predict(boot_matrix)
        boot_labels = _relabel_bootstrap(reference_centers, model.cluster_centers_, boot_labels)
        ref_subset = reference_labels[unique_idx]
        rows.append(
            {
                "bootstrap_iteration": iteration,
                "sample_size": int(len(unique_idx)),
                "adjusted_rand_index": adjusted_rand_score(ref_subset, boot_labels),
            }
        )
    return pd.DataFrame(rows)


def build_review_topic_vectors(doc_info: pd.DataFrame) -> pd.DataFrame:
    counts = (
        doc_info.groupby(["Person_id", "Semantic_Label"]).size().unstack(fill_value=0).reindex(columns=ZERO_SHOT_MAJOR_TOPICS, fill_value=0)
    )
    counts.index = counts.index.astype("Int64")
    counts = counts.reset_index()
    counts["Topic_Total"] = counts[ZERO_SHOT_MAJOR_TOPICS].sum(axis=1)
    props = counts.copy()
    props[ZERO_SHOT_MAJOR_TOPICS] = props[ZERO_SHOT_MAJOR_TOPICS].div(props["Topic_Total"].replace(0, np.nan), axis=0).fillna(0.0)
    return props


def benjamini_hochberg(p_values: Sequence[float]) -> np.ndarray:
    p_values = np.asarray(p_values, dtype=float)
    if p_values.size == 0:
        return np.array([], dtype=float)
    order = np.argsort(p_values)
    ranked = p_values[order]
    adjusted = ranked * len(ranked) / np.arange(1, len(ranked) + 1)
    adjusted = np.minimum.accumulate(adjusted[::-1])[::-1]
    adjusted = np.clip(adjusted, 0.0, 1.0)
    out = np.empty_like(adjusted)
    out[order] = adjusted
    return out


def _permanova_components(matrix: np.ndarray, groups: Sequence[object]) -> tuple[np.ndarray, np.ndarray, float, float, float, float]:
    matrix = np.asarray(matrix, dtype=float)
    groups = np.asarray(groups)
    unique_groups, inverse = np.unique(groups, return_inverse=True)
    grand_mean = matrix.mean(axis=0)
    total_ss = float(((matrix - grand_mean) ** 2).sum())
    within_ss = 0.0
    for group_id in np.unique(inverse):
        subset = matrix[inverse == group_id]
        centroid = subset.mean(axis=0)
        within_ss += float(((subset - centroid) ** 2).sum())
    between_ss = total_ss - within_ss
    numerator_df = len(unique_groups) - 1
    denominator_df = len(matrix) - len(unique_groups)
    if numerator_df <= 0 or denominator_df <= 0 or within_ss == 0.0:
        f_stat = np.nan
    else:
        f_stat = (between_ss / numerator_df) / (within_ss / denominator_df)
    return unique_groups, inverse, total_ss, between_ss, within_ss, f_stat


def permanova_euclidean(matrix: np.ndarray, groups: Sequence[object], n_permutations: int = 999, random_state: int = 42) -> pd.DataFrame:
    unique_groups, inverse, total_ss, between_ss, within_ss, observed_f = _permanova_components(matrix, groups)

    def compute_f(label_index: np.ndarray) -> float:
        within_perm = 0.0
        for group_id in np.unique(label_index):
            subset = matrix[label_index == group_id]
            centroid = subset.mean(axis=0)
            within_perm += float(((subset - centroid) ** 2).sum())
        between_perm = total_ss - within_perm
        numerator_df = len(unique_groups) - 1
        denominator_df = len(matrix) - len(unique_groups)
        if numerator_df <= 0 or denominator_df <= 0 or within_perm == 0.0:
            return np.nan
        return (between_perm / numerator_df) / (within_perm / denominator_df)

    rng = np.random.default_rng(random_state)
    permuted_f = []
    for _ in range(n_permutations):
        permuted_f.append(compute_f(rng.permutation(inverse)))
    permuted_f = np.asarray(permuted_f, dtype=float)
    p_value = (1 + np.sum(permuted_f >= observed_f)) / (1 + len(permuted_f)) if len(permuted_f) else np.nan
    return pd.DataFrame(
        [
            {
                "distance_metric": "euclidean",
                "n_samples": int(len(matrix)),
                "n_groups": int(len(unique_groups)),
                "n_permutations": int(n_permutations),
                "pseudo_f": observed_f,
                "p_value": p_value,
                "r_squared": between_ss / total_ss if total_ss else np.nan,
                "between_ss": between_ss,
                "within_ss": within_ss,
            }
        ]
    )


def topic_separation_vs_null(
    topic_matrix: np.ndarray,
    groups: Sequence[object],
    n_permutations: int = 999,
    random_state: int = 42,
) -> pd.DataFrame:
    groups = np.asarray(groups)
    _, _, total_ss, between_ss, _, observed_f = _permanova_components(topic_matrix, groups)
    observed_r_squared = between_ss / total_ss if total_ss else np.nan

    rng = np.random.default_rng(random_state)
    null_f = []
    null_r_squared = []
    for _ in range(n_permutations):
        _, _, _, perm_between_ss, _, perm_f = _permanova_components(topic_matrix, rng.permutation(groups))
        null_f.append(perm_f)
        null_r_squared.append(perm_between_ss / total_ss if total_ss else np.nan)

    null_f = np.asarray(null_f, dtype=float)
    null_r_squared = np.asarray(null_r_squared, dtype=float)
    return pd.DataFrame(
        [
            {
                "observed_pseudo_f": observed_f,
                "observed_r_squared": observed_r_squared,
                "null_mean_pseudo_f": float(np.nanmean(null_f)) if len(null_f) else np.nan,
                "null_pseudo_f_p95": float(np.nanpercentile(null_f, 95)) if len(null_f) else np.nan,
                "pseudo_f_empirical_p": float((1 + np.sum(null_f >= observed_f)) / (1 + len(null_f))) if len(null_f) else np.nan,
                "observed_minus_null_mean_pseudo_f": float(observed_f - np.nanmean(null_f)) if len(null_f) else np.nan,
                "null_mean_r_squared": float(np.nanmean(null_r_squared)) if len(null_r_squared) else np.nan,
                "null_r_squared_p95": float(np.nanpercentile(null_r_squared, 95)) if len(null_r_squared) else np.nan,
                "r_squared_empirical_p": float((1 + np.sum(null_r_squared >= observed_r_squared)) / (1 + len(null_r_squared))) if len(null_r_squared) else np.nan,
                "observed_minus_null_mean_r_squared": float(observed_r_squared - np.nanmean(null_r_squared)) if len(null_r_squared) else np.nan,
            }
        ]
    )


def benchmark_cluster_feature_sets(
    metadata: pd.DataFrame,
    feature_sets: Mapping[str, np.ndarray],
    topic_vectors: Optional[pd.DataFrame] = None,
    k_values: Iterable[int] = range(2, 9),
    min_cluster_share: float = 0.05,
    random_state: int = 42,
    topic_permutations: int = 199,
    null_permutations: int = 199,
) -> pd.DataFrame:
    topic_matrix = None
    if topic_vectors is not None:
        aligned_topics = metadata[["Person_id"]].merge(
            topic_vectors[["Person_id"] + ZERO_SHOT_MAJOR_TOPICS],
            on="Person_id",
            how="left",
            validate="one_to_one",
            sort=False,
        )
        if aligned_topics[ZERO_SHOT_MAJOR_TOPICS].isna().any().any():
            missing = int(aligned_topics[ZERO_SHOT_MAJOR_TOPICS].isna().any(axis=1).sum())
            raise ValueError(f"Topic vectors are missing for {missing} metadata rows.")
        topic_matrix = aligned_topics[ZERO_SHOT_MAJOR_TOPICS].to_numpy()

    rows = []
    for feature_set_name, matrix in feature_sets.items():
        if matrix.shape[0] != len(metadata):
            raise ValueError(f"Feature set '{feature_set_name}' does not align with metadata row count.")
        metrics = select_cluster_solution(
            matrix,
            k_values=k_values,
            min_cluster_share=min_cluster_share,
            random_state=random_state,
        )
        selected = metrics.loc[metrics["selected"]].iloc[0]
        labels, _ = reorder_clusters(selected["labels"], selected["model"].cluster_centers_)
        counts = pd.Series(labels + 1).value_counts().sort_index()
        row = {
            "feature_set": feature_set_name,
            "k": int(selected["k"]),
            "silhouette_mean": float(selected["silhouette_mean"]),
            "silhouette_sd": float(selected["silhouette_sd"]),
            "calinski_harabasz": float(selected["calinski_harabasz"]),
            "davies_bouldin": float(selected["davies_bouldin"]),
            "min_cluster_size": int(counts.min()),
            "min_cluster_share": float(counts.min() / len(labels)),
        }
        if topic_matrix is not None:
            permanova = permanova_euclidean(
                topic_matrix,
                labels + 1,
                n_permutations=topic_permutations,
                random_state=random_state,
            ).iloc[0]
            null_eval = topic_separation_vs_null(
                topic_matrix,
                labels + 1,
                n_permutations=null_permutations,
                random_state=random_state,
            ).iloc[0]
            row.update({f"topic_{key}": value for key, value in permanova.to_dict().items()})
            row.update({f"topic_null_{key}": value for key, value in null_eval.to_dict().items()})
        rows.append(row)
    return pd.DataFrame(rows).sort_values(["silhouette_mean", "min_cluster_share"], ascending=[False, False]).reset_index(drop=True)


def _kruskal_effect_size(h_stat: float, n_samples: int, n_groups: int) -> float:
    denom = n_samples - n_groups
    return np.nan if denom <= 0 else max(0.0, (h_stat - n_groups + 1) / denom)


def _rank_biserial(u_stat: float, n_a: int, n_b: int) -> float:
    return (2 * u_stat / (n_a * n_b)) - 1


def topic_followup_tests(merged_df: pd.DataFrame, alpha: float = 0.05) -> tuple[pd.DataFrame, pd.DataFrame]:
    cluster_ids = sorted(merged_df["Cluster_ID"].unique())
    omnibus_rows = []
    pairwise_frames = []
    for topic in ZERO_SHOT_MAJOR_TOPICS:
        grouped = [merged_df.loc[merged_df["Cluster_ID"] == cid, topic].to_numpy() for cid in cluster_ids]
        h_stat, p_value = kruskal(*grouped)
        row = {
            "Topic": topic,
            "H_Statistic": h_stat,
            "P_Value": p_value,
            "Effect_Size_Epsilon_Squared": _kruskal_effect_size(h_stat, len(merged_df), len(cluster_ids)),
        }
        for cid in cluster_ids:
            sample = merged_df.loc[merged_df["Cluster_ID"] == cid, topic]
            row[f"Cluster_{cid}_Median"] = float(sample.median())
            row[f"Cluster_{cid}_Mean"] = float(sample.mean())
        omnibus_rows.append(row)
    omnibus_df = pd.DataFrame(omnibus_rows)
    omnibus_df["P_Value_FDR"] = benjamini_hochberg(omnibus_df["P_Value"].to_numpy())
    omnibus_df["Significant_FDR"] = omnibus_df["P_Value_FDR"] < alpha

    for row in omnibus_df.itertuples(index=False):
        if not row.Significant_FDR:
            continue
        topic_rows = []
        for cluster_a, cluster_b in itertools.combinations(cluster_ids, 2):
            sample_a = merged_df.loc[merged_df["Cluster_ID"] == cluster_a, row.Topic].to_numpy()
            sample_b = merged_df.loc[merged_df["Cluster_ID"] == cluster_b, row.Topic].to_numpy()
            u_stat, p_value = mannwhitneyu(sample_a, sample_b, alternative="two-sided", method="asymptotic")
            topic_rows.append(
                {
                    "Topic": row.Topic,
                    "Cluster_A": cluster_a,
                    "Cluster_B": cluster_b,
                    "Median_A": float(np.median(sample_a)),
                    "Median_B": float(np.median(sample_b)),
                    "Mean_A": float(np.mean(sample_a)),
                    "Mean_B": float(np.mean(sample_b)),
                    "U_Statistic": u_stat,
                    "P_Value": p_value,
                    "Rank_Biserial_A_vs_B": _rank_biserial(u_stat, len(sample_a), len(sample_b)),
                }
            )
        topic_df = pd.DataFrame(topic_rows)
        topic_df["P_Value_FDR"] = benjamini_hochberg(topic_df["P_Value"].to_numpy())
        topic_df["Significant_FDR"] = topic_df["P_Value_FDR"] < alpha
        pairwise_frames.append(topic_df)

    pairwise_df = pd.concat(pairwise_frames, ignore_index=True) if pairwise_frames else pd.DataFrame()
    return omnibus_df, pairwise_df


def optional_cluster_regressions(merged_df: pd.DataFrame) -> Optional[pd.DataFrame]:
    try:
        import statsmodels.api as sm
        import statsmodels.formula.api as smf
    except ImportError:
        return None

    df = merged_df.copy()
    df["Cluster"] = df["Cluster"].astype("category")
    df["City"] = df["City"].astype("category")
    df["Season"] = df["Season"].astype("category")
    df["Reviewer_Score"] = pd.to_numeric(df["Reviewer_Score"], errors="coerce")
    rows = []
    for topic in ZERO_SHOT_MAJOR_TOPICS:
        df["topic_present"] = (df[topic] > 0).astype(int)
        try:
            model = smf.glm(
                "topic_present ~ C(Cluster) + C(City) + C(Season) + Reviewer_Score",
                data=df.dropna(subset=["Reviewer_Score", "Cluster", "City", "Season"]),
                family=sm.families.Binomial(),
            ).fit()
        except Exception:
            continue
        conf_int = model.conf_int()
        for term, coefficient in model.params.items():
            if not term.startswith("C(Cluster)"):
                continue
            rows.append(
                {
                    "Topic": topic,
                    "Term": term,
                    "Coefficient": coefficient,
                    "Odds_Ratio": float(np.exp(coefficient)),
                    "P_Value": model.pvalues.get(term, np.nan),
                    "CI_Lower": float(np.exp(conf_int.loc[term, 0])),
                    "CI_Upper": float(np.exp(conf_int.loc[term, 1])),
                    "Model_N": int(model.nobs),
                }
            )
    if not rows:
        return pd.DataFrame()
    out = pd.DataFrame(rows)
    out["P_Value_FDR"] = benjamini_hochberg(out["P_Value"].to_numpy())
    return out
