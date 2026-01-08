import networkx as nx
from tqdm.notebook import tqdm

def build_citation_graph(queries, docs):
    # Fusionner en un seul dictionnaire
    all_docs = {**queries, **docs}

    G = nx.DiGraph()

    # Ajouter tous les nœuds
    for doc_id, content in all_docs.items():
        G.add_node(doc_id, **content["metadata"])

    # Ajouter les arêtes de citation
    for doc_id, content in all_docs.items():
        references = content["metadata"].get("references", [])
        
        for ref_id in references:
            if ref_id in all_docs:    # évite les références manquantes
                G.add_edge(doc_id, ref_id)  # doc_id --> ref_id (doc_id cites ref_id)

    return G


def get_all_ppr(G, query_ids, alpha=0.85):
    ppr_dict = {}
    for query_id in tqdm(query_ids):
        ppr = nx.pagerank(G, alpha=alpha, personalization={query_id: 1.0})
        ppr_dict[query_id] = ppr
    return ppr_dict
