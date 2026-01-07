import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from scipy.sparse import find
from sklearn.metrics import roc_auc_score, roc_curve
from IPython.display import display


def load_corpus(file_path: str) -> dict[str, dict]:
    """
    Load corpus data from JSONL file.
    Returns dictionary mapping document IDs to document data.
    """
    corpus = {}
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            doc = json.loads(line)
            id = doc.pop('_id')
            corpus[id] = doc
    return corpus


def load_queries(file_path: str) -> dict[str, dict]:
    """
    Load query data from JSONL file.
    Returns dictionary mapping query IDs to query data.
    """
    queries = {}
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            query = json.loads(line)
            id = query.pop('_id')
            queries[id] = query
    return queries
    

def load_qrels(file_path: str) -> dict[str, dict[str, int]]:
    """
    Load relevance judgments from TSV file.
    Returns dictionary mapping query IDs to candidate relevance scores.
    """
    qrels = {}
    with open(file_path, 'r', encoding='utf-8') as f:
        next(f)  # Skip header line
        for line in f:
            query_id, doc_id, score = line.strip().split('\t')
            if query_id not in qrels:
                qrels[query_id] = {}
            qrels[query_id][doc_id] = int(score)
    return qrels


def load_test(file_path: str) -> dict[str, dict[str, int]]:
    """
    Load test relevance judgments from TSV file.
    """
    test_set = {}
    with open(file_path, 'r', encoding='utf-8') as f:
        next(f)  # Skip header line
        for line in f:
            query_id, doc_id, _ = line.strip().split('\t')
            if query_id not in test_set:
                test_set[query_id] = {}
            test_set[query_id][doc_id] = "#"
    return test_set


def preprocessing(corpus: dict[str, dict], queries: dict[str, dict]) -> tuple[dict[str, str], dict[str, str]]:
    """
    Merge title and text for documents in corpus and extract text for queries.
    """
    preprocessed_corpus = {}
    preprocessed_queries = {}
    for doc_id, doc in corpus.items():
        preprocessed_corpus[doc_id] = f"{doc.get('title', '')}\n{doc.get('text', '')}"
    for query_id, query in queries.items():
        preprocessed_queries[query_id] = f"{query.get('text', '')}"
    return preprocessed_corpus, preprocessed_queries


def print_feats(v, features, top_n = 30):
    """
    Extract and display the top N features from a sparse vector.
    """
    _, ids, values = find(v)
    feats = [(ids[i], values[i], features[ids[i]]) for i in range(len(list(ids)))]
    top_feats = sorted(feats, key=lambda x: x[1], reverse=True)[0:top_n]
    return pd.DataFrame({"word" : [t[2] for t in top_feats], "value": [t[1] for t in top_feats]}) 


def save_embeddings(encoder, texts: dict[str, str], file_path: str, device='cpu'):
    """
    Encode texts and save embeddings to a .npz file."""
    ids = np.array(list(texts.keys()))
    vectors = encoder.encode(list(texts.values()), device=device, convert_to_numpy=True, show_progress_bar=True)
    np.savez(file_path, ids=ids, vectors=vectors)


def load_embeddings(file_path: str) -> dict[str, np.ndarray]:
    """
    Load embeddings from a .npz file.
    Returns a dictionary mapping IDs to their corresponding vectors.
    """
    data = np.load(file_path)
    embeddings = {str(id): vector for id, vector in zip(data['ids'], data['vectors'])}
    return embeddings


def plot_auroc(y_true: np.ndarray, y_pred: np.ndarray):

    auc = roc_auc_score(y_true, y_pred)
    fpr, tpr, thresholds = roc_curve(y_true, y_pred)

    # Youden's J statistic
    J = tpr - fpr
    best_idx = np.argmax(J)
    best_threshold = thresholds[best_idx]

    plt.figure()
    plt.plot(fpr, tpr, label=f'ROC curve (area = {auc:.2f})')
    plt.scatter(fpr[best_idx], tpr[best_idx], s=100, c='red', label=f'Best threshold: {best_threshold:.2f}')
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic')
    plt.legend(loc="lower right")
    plt.show()

    return best_threshold


def display_metrics(metrics: dict, decimals: int = 3):
    names = {
        "precision": "Precision",
        "recall": "Recall",
        "f1_score": "F1-score",
        "auroc": "AUROC",
    }

    df = (
        pd.DataFrame({
            "Metric": [names.get(k, k) for k in metrics.keys()],
            "Score": [f"{v:.{decimals}f}" for v in metrics.values()]
        })
        .rename(columns=names)
    )
    display(df.style.hide(axis="index"))


def save_predictions(preds: dict[str, dict[str, int]], test_set: dict[str, dict[str, int]], file_path: str):
    with open(file_path, 'w', encoding='utf-8') as f:
        f.write("RowId,query-id,corpus-id,score\n")
        row_id = 1
        for query_id in test_set.keys():
            for doc_id in test_set[query_id].keys():
                score = preds[query_id].get(doc_id, 0)
                f.write(f"{row_id},{query_id},{doc_id},{score}\n")
                row_id += 1