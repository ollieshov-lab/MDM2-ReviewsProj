from __future__ import annotations

import copy
import json
import math
import re
from collections import Counter
from pathlib import Path
import sys
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, f1_score, precision_recall_fscore_support
from sklearn.model_selection import GroupKFold, GroupShuffleSplit
from sklearn.multiclass import OneVsRestClassifier
from sklearn.preprocessing import MultiLabelBinarizer

CURRENT_DIR = Path(__file__).resolve().parent
LIB_DIR = CURRENT_DIR / "lib"
if str(LIB_DIR) not in sys.path:
    sys.path.insert(0, str(LIB_DIR))

from rp_paths import resolve_hotel_document_info_csv, resolve_stratified_reviews_csv

from guest_clusters_helpers import ZERO_SHOT_MAJOR_TOPICS, find_repo_root, normalize_tag, parse_tags

REVIEW_COLUMNS = ["ID", "Hotel_Name", "City", "Review_Date", "Tags"]
STRUCTURAL_TAG_VALUES = {
    "business trip",
    "leisure trip",
    "couple",
    "group",
    "solo traveler",
    "solo traveller",
    "single",
    "family with young children",
    "family with older children",
    "submitted from a mobile device",
    "with a pet",
}


def classify_tag_group(tag: str) -> str:
    normalized = normalize_tag(tag)
    if normalized in STRUCTURAL_TAG_VALUES:
        return "structural"
    if re.fullmatch(r"stayed\s+\d+\s+nights?", normalized):
        return "structural"
    if re.fullmatch(r"\d+\s+rooms?", normalized):
        return "structural"
    return "room_type"


def _dedupe_preserve_order(values: Sequence[str]) -> list[str]:
    return list(dict.fromkeys(values))


def _normalize_tag_list(tags_value: object) -> list[str]:
    raw_tags = parse_tags(tags_value)
    normalized = [normalize_tag(tag) for tag in raw_tags if normalize_tag(tag)]
    return _dedupe_preserve_order(normalized)


def load_review_topic_dataset(repo_root: Path | None = None) -> pd.DataFrame:
    repo_root = find_repo_root(repo_root)
    review_path = resolve_stratified_reviews_csv()
    doc_info_path = resolve_hotel_document_info_csv()

    review_df = pd.read_csv(review_path, usecols=REVIEW_COLUMNS).rename(columns={"ID": "Person_id"})
    review_df["Person_id"] = pd.to_numeric(review_df["Person_id"], errors="coerce").astype("Int64")
    review_df["Review_Date"] = pd.to_datetime(review_df["Review_Date"], errors="coerce")
    review_df = review_df.dropna(subset=["Person_id"]).drop_duplicates(subset=["Person_id"]).copy()
    review_df["Parsed_Tags"] = review_df["Tags"].apply(parse_tags)
    review_df["Normalized_Tags"] = review_df["Tags"].apply(_normalize_tag_list)
    review_df["Structural_Tags"] = review_df["Normalized_Tags"].apply(
        lambda tags: [tag for tag in tags if classify_tag_group(tag) == "structural"]
    )
    review_df["Room_Tags"] = review_df["Normalized_Tags"].apply(
        lambda tags: [tag for tag in tags if classify_tag_group(tag) == "room_type"]
    )
    review_df["Tag_Flag_Count"] = review_df["Normalized_Tags"].apply(len)

    doc_info = pd.read_csv(doc_info_path, usecols=["Person_id", "Semantic_Label"])
    doc_info["Person_id"] = pd.to_numeric(doc_info["Person_id"], errors="coerce").astype("Int64")
    doc_info = doc_info.dropna(subset=["Person_id"]).copy()
    doc_info = doc_info[doc_info["Semantic_Label"].isin(ZERO_SHOT_MAJOR_TOPICS)].copy()

    topic_counts = (
        doc_info.groupby(["Person_id", "Semantic_Label"])
        .size()
        .unstack(fill_value=0)
        .reindex(columns=ZERO_SHOT_MAJOR_TOPICS, fill_value=0)
    )
    topic_counts.index = topic_counts.index.astype("Int64")
    topic_counts = topic_counts.reset_index()
    topic_counts["Topic_Total"] = topic_counts[ZERO_SHOT_MAJOR_TOPICS].sum(axis=1)
    topic_counts["Active_Topic_Count"] = (topic_counts[ZERO_SHOT_MAJOR_TOPICS] > 0).sum(axis=1)

    topic_proportions = topic_counts[["Person_id", "Topic_Total", "Active_Topic_Count"]].copy()
    topic_proportions[ZERO_SHOT_MAJOR_TOPICS] = (
        topic_counts[ZERO_SHOT_MAJOR_TOPICS]
        .div(topic_counts["Topic_Total"].replace(0, np.nan), axis=0)
        .fillna(0.0)
    )
    for topic in ZERO_SHOT_MAJOR_TOPICS:
        topic_proportions[f"Count__{topic}"] = topic_counts[topic].to_numpy()
        topic_proportions[f"Has__{topic}"] = (topic_counts[topic] > 0).astype(int).to_numpy()

    dataset = review_df.merge(topic_proportions, on="Person_id", how="inner", validate="one_to_one")
    dataset = dataset.sort_values(["Hotel_Name", "Person_id"]).reset_index(drop=True)
    return dataset


def dataset_summary_table(dataset: pd.DataFrame) -> pd.DataFrame:
    rows = [
        {"Metric": "rows", "Value": int(len(dataset))},
        {"Metric": "hotels", "Value": int(dataset["Hotel_Name"].nunique())},
        {"Metric": "cities", "Value": int(dataset["City"].nunique())},
        {"Metric": "mean_tags_per_review", "Value": float(dataset["Tag_Flag_Count"].mean())},
        {"Metric": "mean_active_topics", "Value": float(dataset["Active_Topic_Count"].mean())},
        {"Metric": "single_topic_share", "Value": float((dataset["Active_Topic_Count"] == 1).mean())},
        {"Metric": "two_topic_share", "Value": float((dataset["Active_Topic_Count"] == 2).mean())},
    ]
    for topic in ZERO_SHOT_MAJOR_TOPICS:
        rows.append({"Metric": f"share_with__{topic}", "Value": float(dataset[f"Has__{topic}"].mean())})
    return pd.DataFrame(rows)


def build_target_matrix(dataset: pd.DataFrame) -> np.ndarray:
    return dataset[[f"Has__{topic}" for topic in ZERO_SHOT_MAJOR_TOPICS]].to_numpy(dtype=np.int64)


def group_train_val_test_split(
    dataset: pd.DataFrame,
    group_col: str = "Hotel_Name",
    train_size: float = 0.64,
    val_size: float = 0.16,
    test_size: float = 0.20,
    random_state: int = 42,
) -> tuple[dict[str, pd.DataFrame], pd.DataFrame]:
    total = train_size + val_size + test_size
    if not math.isclose(total, 1.0, rel_tol=1e-9, abs_tol=1e-9):
        raise ValueError("train_size, val_size, and test_size must sum to 1.0")

    groups = dataset[group_col].astype(str)
    splitter = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=random_state)
    train_val_idx, test_idx = next(splitter.split(dataset, groups=groups))

    train_val_df = dataset.iloc[train_val_idx].reset_index(drop=True)
    test_df = dataset.iloc[test_idx].reset_index(drop=True)

    val_fraction_of_train_val = val_size / (train_size + val_size)
    splitter = GroupShuffleSplit(n_splits=1, test_size=val_fraction_of_train_val, random_state=random_state + 1)
    train_idx, val_idx = next(splitter.split(train_val_df, groups=train_val_df[group_col].astype(str)))

    train_df = train_val_df.iloc[train_idx].reset_index(drop=True)
    val_df = train_val_df.iloc[val_idx].reset_index(drop=True)

    splits = {"train": train_df, "validation": val_df, "test": test_df}
    rows = []
    for split_name, split_df in splits.items():
        rows.append(
            {
                "Split": split_name,
                "Rows": int(len(split_df)),
                "Hotels": int(split_df[group_col].nunique()),
                "Cities": int(split_df["City"].nunique()),
                "Mean_Active_Topics": float(split_df["Active_Topic_Count"].mean()),
            }
        )
    summary = pd.DataFrame(rows)
    return splits, summary


def build_tag_matrices(
    train_df: pd.DataFrame,
    validation_df: pd.DataFrame,
    test_df: pd.DataFrame,
    tag_column: str = "Normalized_Tags",
    min_df: int = 5,
) -> tuple[dict[str, Any], pd.DataFrame]:
    train_tags = train_df[tag_column].tolist()
    frequency = Counter(tag for tags in train_tags for tag in set(tags))
    vocabulary = sorted(tag for tag, count in frequency.items() if count >= min_df)
    if not vocabulary:
        raise ValueError(f"No tags from column '{tag_column}' met min_df={min_df}.")

    mlb = MultiLabelBinarizer(classes=vocabulary, sparse_output=True)
    mlb.fit([[]])

    matrices = {
        "train": mlb.transform(train_df[tag_column]),
        "validation": mlb.transform(validation_df[tag_column]),
        "test": mlb.transform(test_df[tag_column]),
        "binarizer": mlb,
        "tag_column": tag_column,
    }

    vocabulary_df = pd.DataFrame(
        {
            "Normalized_Tag": vocabulary,
            "Document_Frequency_Train": [int(frequency[tag]) for tag in vocabulary],
            "Share_of_Train_Reviews": [float(frequency[tag] / len(train_df)) for tag in vocabulary],
            "Feature_Group": [classify_tag_group(tag) for tag in vocabulary],
        }
    )
    return matrices, vocabulary_df


def make_feature_set_summary(
    dataset: pd.DataFrame,
    splits: Mapping[str, pd.DataFrame],
    feature_specs: Mapping[str, str],
    min_df: int = 5,
) -> pd.DataFrame:
    rows = []
    for feature_name, tag_column in feature_specs.items():
        matrices, vocabulary_df = build_tag_matrices(
            splits["train"],
            splits["validation"],
            splits["test"],
            tag_column=tag_column,
            min_df=min_df,
        )
        rows.append(
            {
                "Feature_Set": feature_name,
                "Tag_Column": tag_column,
                "Train_Features": int(matrices["train"].shape[1]),
                "Mean_Tags_Per_Review": float(dataset[tag_column].apply(len).mean()),
                "Structured_Features": int((vocabulary_df["Feature_Group"] == "structural").sum()),
                "Room_Features": int((vocabulary_df["Feature_Group"] == "room_type").sum()),
            }
        )
    return pd.DataFrame(rows)


def build_topk_frequency_baseline(train_y: np.ndarray, n_samples: int, top_k: int | None = None) -> tuple[np.ndarray, np.ndarray]:
    prevalence = train_y.mean(axis=0)
    if top_k is None:
        top_k = max(1, int(round(train_y.sum(axis=1).mean())))
    ranking = np.argsort(-prevalence)[:top_k]
    probabilities = np.tile(prevalence, (n_samples, 1))
    predictions = np.zeros_like(probabilities, dtype=np.int64)
    predictions[:, ranking] = 1
    return probabilities, predictions


def fit_logistic_baseline(X_train: Any, y_train: np.ndarray) -> OneVsRestClassifier:
    estimator = LogisticRegression(
        max_iter=1000,
        class_weight="balanced",
        solver="liblinear",
    )
    model = OneVsRestClassifier(estimator)
    model.fit(X_train, y_train)
    return model


def predict_logistic_probabilities(model: OneVsRestClassifier, matrix: Any) -> np.ndarray:
    probabilities = model.predict_proba(matrix)
    return np.asarray(probabilities, dtype=np.float32)


def choose_thresholds(y_true: np.ndarray, y_prob: np.ndarray, thresholds: Sequence[float] | None = None) -> np.ndarray:
    if thresholds is None:
        thresholds = np.arange(0.1, 0.91, 0.05)
    thresholds = np.asarray(list(thresholds), dtype=np.float32)
    best_thresholds = []
    for idx in range(y_true.shape[1]):
        if y_true[:, idx].sum() == 0:
            best_thresholds.append(0.5)
            continue
        best_threshold = 0.5
        best_score = -1.0
        for threshold in thresholds:
            preds = (y_prob[:, idx] >= threshold).astype(np.int64)
            score = f1_score(y_true[:, idx], preds, zero_division=0)
            if score > best_score or (math.isclose(score, best_score) and abs(threshold - 0.5) < abs(best_threshold - 0.5)):
                best_score = score
                best_threshold = float(threshold)
        best_thresholds.append(best_threshold)
    return np.asarray(best_thresholds, dtype=np.float32)


def evaluate_multilabel_predictions(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    topics: Sequence[str],
    model_name: str,
    split_name: str,
    thresholds: Sequence[float] | None = None,
    predictions_override: np.ndarray | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if predictions_override is None:
        if thresholds is None:
            thresholds = np.full(y_true.shape[1], 0.5, dtype=np.float32)
        predictions = (y_prob >= np.asarray(thresholds, dtype=np.float32)[None, :]).astype(np.int64)
    else:
        predictions = predictions_override.astype(np.int64)
        if thresholds is None:
            thresholds = np.full(y_true.shape[1], np.nan, dtype=np.float32)

    macro_f1 = f1_score(y_true, predictions, average="macro", zero_division=0)
    micro_f1 = f1_score(y_true, predictions, average="micro", zero_division=0)
    exact_match = float((predictions == y_true).all(axis=1).mean())
    top1_idx = y_prob.argmax(axis=1)
    top1_accuracy = float(np.mean(y_true[np.arange(len(y_true)), top1_idx] > 0))
    top2_idx = np.argsort(-y_prob, axis=1)[:, :2]
    top2_hit_rate = float(np.mean([bool(y_true[row_idx, idx].any()) for row_idx, idx in enumerate(top2_idx)]))

    overall_metrics = pd.DataFrame(
        [
            {
                "Model": model_name,
                "Split": split_name,
                "Samples": int(len(y_true)),
                "Topics": int(len(topics)),
                "Macro_F1": float(macro_f1),
                "Micro_F1": float(micro_f1),
                "Exact_Match": exact_match,
                "Top1_Accuracy": top1_accuracy,
                "Top2_Hit_Rate": top2_hit_rate,
                "Mean_True_Active_Topics": float(y_true.sum(axis=1).mean()),
                "Mean_Predicted_Active_Topics": float(predictions.sum(axis=1).mean()),
                "Share_With_Any_Prediction": float((predictions.sum(axis=1) > 0).mean()),
            }
        ]
    )

    precision, recall, f1_values, support = precision_recall_fscore_support(
        y_true,
        predictions,
        average=None,
        zero_division=0,
    )
    rows = []
    for idx, topic in enumerate(topics):
        if np.unique(y_true[:, idx]).size < 2:
            pr_auc = np.nan
        else:
            pr_auc = float(average_precision_score(y_true[:, idx], y_prob[:, idx]))
        rows.append(
            {
                "Model": model_name,
                "Split": split_name,
                "Topic": topic,
                "Support": int(support[idx]),
                "Threshold": float(thresholds[idx]) if not np.isnan(thresholds[idx]) else np.nan,
                "Precision": float(precision[idx]),
                "Recall": float(recall[idx]),
                "F1": float(f1_values[idx]),
                "PR_AUC": pr_auc,
                "Predicted_Positive_Rate": float(predictions[:, idx].mean()),
                "Actual_Positive_Rate": float(y_true[:, idx].mean()),
            }
        )
    per_topic_metrics = pd.DataFrame(rows)
    return overall_metrics, per_topic_metrics


def make_prediction_frame(
    frame: pd.DataFrame,
    y_true: np.ndarray,
    y_prob: np.ndarray,
    topics: Sequence[str],
    thresholds: Sequence[float] | None,
    model_name: str,
    split_name: str,
    predictions_override: np.ndarray | None = None,
) -> pd.DataFrame:
    if predictions_override is None:
        if thresholds is None:
            thresholds = np.full(y_true.shape[1], 0.5, dtype=np.float32)
        y_pred = (y_prob >= np.asarray(thresholds, dtype=np.float32)[None, :]).astype(np.int64)
    else:
        y_pred = predictions_override.astype(np.int64)
    out = frame[["Person_id", "Hotel_Name", "City", "Review_Date", "Tag_Flag_Count", "Tags"]].copy()
    out.insert(0, "Split", split_name)
    out.insert(0, "Model", model_name)
    out["True_Active_Topic_Count"] = y_true.sum(axis=1)
    out["Predicted_Active_Topic_Count"] = y_pred.sum(axis=1)
    for idx, topic in enumerate(topics):
        safe = topic.replace(" ", "_").replace("&", "and").replace("/", "_")
        out[f"actual__{safe}"] = y_true[:, idx]
        out[f"predicted__{safe}"] = y_pred[:, idx]
        out[f"probability__{safe}"] = y_prob[:, idx]
    return out


class TopicPresenceMLP(nn.Module):
    def __init__(self, input_dim: int, output_dim: int, hidden_dims: Sequence[int] = (256, 128), dropout: float = 0.2):
        super().__init__()
        dims = [input_dim, *hidden_dims, output_dim]
        layers: list[nn.Module] = []
        for idx in range(len(dims) - 2):
            layers.append(nn.Linear(dims[idx], dims[idx + 1]))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
        layers.append(nn.Linear(dims[-2], dims[-1]))
        self.network = nn.Sequential(*layers)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.network(features)


def _dense_float32(matrix: Any) -> np.ndarray:
    if hasattr(matrix, "toarray"):
        matrix = matrix.toarray()
    return np.asarray(matrix, dtype=np.float32)


def train_mlp(
    X_train: Any,
    y_train: np.ndarray,
    X_validation: Any,
    y_validation: np.ndarray,
    hidden_dims: Sequence[int] = (256, 128),
    dropout: float = 0.2,
    learning_rate: float = 1e-3,
    weight_decay: float = 1e-4,
    batch_size: int = 256,
    max_epochs: int = 50,
    patience: int = 10,
    min_delta: float = 1e-4,
    random_state: int = 42,
    device: str | None = None,
) -> tuple[TopicPresenceMLP, pd.DataFrame, dict[str, Any]]:
    torch.manual_seed(random_state)
    np.random.seed(random_state)

    train_x = _dense_float32(X_train)
    train_y = np.asarray(y_train, dtype=np.float32)
    validation_x = _dense_float32(X_validation)
    validation_y = np.asarray(y_validation, dtype=np.float32)

    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    model = TopicPresenceMLP(
        input_dim=train_x.shape[1],
        output_dim=train_y.shape[1],
        hidden_dims=hidden_dims,
        dropout=dropout,
    ).to(device)

    positive_counts = train_y.sum(axis=0)
    negative_counts = train_y.shape[0] - positive_counts
    pos_weight = np.divide(
        negative_counts,
        np.maximum(positive_counts, 1.0),
        out=np.ones_like(negative_counts, dtype=np.float32),
        where=np.maximum(positive_counts, 1.0) > 0,
    )
    criterion = nn.BCEWithLogitsLoss(pos_weight=torch.as_tensor(pos_weight, dtype=torch.float32, device=device))
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

    dataset = torch.utils.data.TensorDataset(
        torch.from_numpy(train_x),
        torch.from_numpy(train_y),
    )
    generator = torch.Generator()
    generator.manual_seed(random_state)
    train_loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        generator=generator,
    )

    validation_x_tensor = torch.from_numpy(validation_x).to(device)
    validation_y_tensor = torch.from_numpy(validation_y).to(device)

    best_state = copy.deepcopy(model.state_dict())
    best_epoch = 0
    best_val_loss = float("inf")
    wait = 0
    history_rows = []

    for epoch in range(1, max_epochs + 1):
        model.train()
        batch_losses = []
        for batch_x, batch_y in train_loader:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)
            optimizer.zero_grad()
            logits = model(batch_x)
            loss = criterion(logits, batch_y)
            loss.backward()
            optimizer.step()
            batch_losses.append(float(loss.detach().cpu().item()))

        model.eval()
        with torch.no_grad():
            validation_logits = model(validation_x_tensor)
            validation_loss = float(criterion(validation_logits, validation_y_tensor).detach().cpu().item())
            validation_prob = torch.sigmoid(validation_logits).detach().cpu().numpy()
        validation_f1 = f1_score(y_validation, (validation_prob >= 0.5).astype(np.int64), average="macro", zero_division=0)
        train_loss = float(np.mean(batch_losses)) if batch_losses else np.nan
        is_best_epoch = validation_loss < (best_val_loss - min_delta)
        if is_best_epoch:
            best_state = copy.deepcopy(model.state_dict())
            best_epoch = epoch
            best_val_loss = validation_loss
            wait = 0
        else:
            wait += 1

        history_rows.append(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "validation_loss": validation_loss,
                "validation_macro_f1_at_0_5": float(validation_f1),
                "is_best_epoch": bool(is_best_epoch),
            }
        )
        if wait >= patience:
            break

    model.load_state_dict(best_state)
    history = pd.DataFrame(history_rows)
    config = {
        "input_dim": int(train_x.shape[1]),
        "output_dim": int(train_y.shape[1]),
        "hidden_dims": list(hidden_dims),
        "dropout": float(dropout),
        "learning_rate": float(learning_rate),
        "weight_decay": float(weight_decay),
        "batch_size": int(batch_size),
        "max_epochs": int(max_epochs),
        "patience": int(patience),
        "min_delta": float(min_delta),
        "best_epoch": int(best_epoch),
        "best_validation_loss": float(best_val_loss),
        "device": device,
    }
    return model, history, config


def predict_mlp_probabilities(model: TopicPresenceMLP, matrix: Any, batch_size: int = 1024, device: str | None = None) -> np.ndarray:
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.eval()
    features = _dense_float32(matrix)
    tensor = torch.from_numpy(features)
    loader = torch.utils.data.DataLoader(tensor, batch_size=batch_size, shuffle=False)
    outputs = []
    with torch.no_grad():
        for batch in loader:
            logits = model(batch.to(device))
            outputs.append(torch.sigmoid(logits).detach().cpu().numpy())
    return np.vstack(outputs)


def run_grouped_logistic_cv(
    dataset: pd.DataFrame,
    tag_column: str = "Normalized_Tags",
    min_df: int = 5,
    n_splits: int = 5,
) -> pd.DataFrame:
    groups = dataset["Hotel_Name"].astype(str)
    gkf = GroupKFold(n_splits=n_splits)
    rows = []
    for fold_idx, (train_idx, validation_idx) in enumerate(gkf.split(dataset, groups=groups), start=1):
        train_df = dataset.iloc[train_idx].reset_index(drop=True)
        validation_df = dataset.iloc[validation_idx].reset_index(drop=True)
        matrices, _ = build_tag_matrices(train_df, validation_df, validation_df, tag_column=tag_column, min_df=min_df)
        y_train = build_target_matrix(train_df)
        y_validation = build_target_matrix(validation_df)
        model = fit_logistic_baseline(matrices["train"], y_train)
        validation_prob = predict_logistic_probabilities(model, matrices["validation"])
        thresholds = choose_thresholds(y_validation, validation_prob)
        overall_metrics, _ = evaluate_multilabel_predictions(
            y_validation,
            validation_prob,
            ZERO_SHOT_MAJOR_TOPICS,
            model_name="logistic_regression",
            split_name=f"cv_fold_{fold_idx}",
            thresholds=thresholds,
        )
        metrics_row = overall_metrics.iloc[0].to_dict()
        metrics_row["Fold"] = fold_idx
        rows.append(metrics_row)
    return pd.DataFrame(rows)


def save_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
