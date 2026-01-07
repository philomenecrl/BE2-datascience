import numpy as np

from tqdm.notebook import tqdm
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score, roc_curve
from sklearn.metrics.pairwise import cosine_similarity


def query_document_similarity(query_vector, document_vectors, top_k: int = 10):
    similarities = cosine_similarity(query_vector, document_vectors)
    similarities = similarities.squeeze()

    # On récupère les top_k meilleurs scores et indices associés
    top_indices = np.argsort(-similarities)[:top_k]
    top_scores = np.take_along_axis(similarities, top_indices, axis=0)
    return top_indices.squeeze(), top_scores.squeeze()


def sparse_browser(
        query: dict[str, str],
        corpus: dict[str, str],
        vectorizer,
        top_k: int = 10,
    ):
    """
    Perform sparse retrieval using vectorizer and return top_k similar documents.
    """

    corpus_ids = list(corpus.keys())

    X_corpus = vectorizer.transform(corpus.values())
    X_query = vectorizer.transform(query.values())
    top_indices, top_scores = query_document_similarity(X_query, X_corpus, top_k=top_k)
    
    results = {corpus_ids[idx]: float(top_scores[i]) for i, idx in enumerate(top_indices)}
    return results


def softmax(x, T=1.0):
    x = np.array(x, dtype=float) / T
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum()


def dense_browser(
    query_embedding: dict[str, np.ndarray],
    corpus_embeddings: dict[str, np.ndarray],
    encoder,
    ppr: dict[str, float] = None,
    beta: float = 0.0,
    temperature: float = 2.0,
    top_k: int = 10,
):  
    """
    Perform dense retrieval with optional PPR re-ranking.
    Combines similarity scores from embeddings with PPR scores if provided.
    """
    
    query_id = list(query_embedding.keys())[0]
    corpus_ids = list(corpus_embeddings.keys())
    
    query_vector = np.array(list(query_embedding.values())[0])
    corpus_vectors = np.array([corpus_embeddings[cid] for cid in corpus_ids])
    
    similarities =  encoder.similarity(query_vector, corpus_vectors)
    similarities = similarities.squeeze().numpy()

    if ppr is not None:
        ppr_scores = np.array([ppr.get(cid, 0.0) for cid in corpus_ids], dtype=float)
        similarities = softmax(similarities, T=temperature)
        ppr_scores = softmax(ppr_scores, T=temperature)
        final_scores = similarities * (1 - beta) + ppr_scores * beta
    else:
        final_scores = similarities
    
    top_indices = np.argsort(-final_scores)[:top_k]
    results = {corpus_ids[idx]: final_scores[idx].item() for idx in top_indices}
    return results


def get_best_threshold(y_true, y_pred):
    """
    Extract the best threshold from ROC curve using Youden's J statistic on logits.
    """
    fpr, tpr, thresholds = roc_curve(y_true, y_pred)

    # Youden's J statistic
    J = tpr - fpr
    best_idx = np.argmax(J)
    return thresholds[best_idx]


def get_logits(
    browser,
    qrels_valid,
    X_queries,
    X_corpus,
    encoder,
    ppr_dict=None,
    beta=0.9,
    temperature=2.0,
    top_k=30,
):
    is_sparse = browser.__name__ == "sparse_browser"

    preds = []
    targets = []

    for query_id in tqdm(qrels_valid.keys()):
        query = {query_id: X_queries[query_id]}
        corpus =  {doc_id: X_corpus[doc_id] for doc_id in qrels_valid[query_id].keys()}
        ppr = ppr_dict[query_id] if ppr_dict is not None else None

        if is_sparse:
            results = browser(query, corpus, encoder, top_k=top_k)
        else:
            results = browser(query, corpus, encoder, ppr=ppr, beta=beta, temperature=temperature, top_k=top_k)
        
        preds.extend(list(dict(sorted(results.items())).values()))
        targets.extend(list(dict(sorted(qrels_valid[query_id].items())).values()))

    y_pred = np.array(preds)
    y_true = np.array(targets)
    return y_pred, y_true


def valid_browser(
        browser,
        qrels_valid,
        X_queries,
        X_corpus,
        encoder,
        ppr_dict=None,
        beta=0.9,
        temperature=2.0,
):
    """
    Validate the browser and compute precision, recall, F1-score, and AUROC.
    """
   
    y_pred, y_true = get_logits(
        browser,
        qrels_valid,
        X_queries,
        X_corpus,
        encoder,
        ppr_dict,
        beta=beta,
        temperature=temperature,
        top_k=30,
    )
    threshold = get_best_threshold(y_true, y_pred)
    y_pred = (y_pred >= threshold).astype(int)

    metrics = {
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred),
        "f1_score": f1_score(y_true, y_pred),
        "auroc": roc_auc_score(y_true, y_pred),
    }
    return metrics


def predict_browser(
    browser,
    test_set,
    X_queries,
    X_corpus,
    encoder,
    threshold: float = 0.5,
    ppr_dict: dict[str, float] = None,
    beta: float = 0.99,
    temperature: float = 3.0,
):  
    preds = {}
    for query_id in tqdm(test_set.keys()):
        query = {query_id: X_queries[query_id]}
        corpus = {doc_id: X_corpus[doc_id] for doc_id in test_set[query_id].keys()}
        ppr = ppr_dict[query_id] if ppr_dict is not None else None

        results = browser(query, corpus, encoder, ppr=ppr, beta=beta, temperature=temperature, top_k=30)
        results = {k: (1 if i < threshold else 0) for i, k in enumerate(results.keys())}
        preds[query_id] = results
    return preds