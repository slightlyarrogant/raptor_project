import time
from typing import Dict, List, Tuple, Optional
import uuid
import numpy as np
import pandas as pd
from langchain.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from src.prepare.embeddings import embed_texts
from src.utils.config import CLUSTERING_DIM, CLUSTERING_THRESHOLD, MAX_RECURSION_LEVELS, VERBOSE, SUMMARIZATION_PROMPT
from src.utils.pinecone_utils import upsert_to_pinecone, query_pinecone
import umap
from sklearn.mixture import GaussianMixture

class RaptorTree:
    def __init__(self, vectorstore, embd, model):
        self.vectorstore = vectorstore
        self.embd = embd
        self.model = model
        self.version = time.time()  # Use timestamp for version control

    def add_new_files(self, new_texts: List[str]):
        if VERBOSE:
            print(f"Adding {len(new_texts)} new texts to the RAPTOR tree")
        
        # Step 1: Embed and cluster the new texts
        df_clusters, df_summary = self.embed_cluster_summarize_texts(new_texts, level=1)
        
        # Step 2: Recursively summarize and build the tree
        tree = self.build_tree(df_summary, level=1)
        
        if VERBOSE:
            print(f"Final tree structure: {tree}")
        
        # Step 3: Update the vector store with the new tree and original documents
        self.update_vector_store(tree, df_clusters)
        self.version = time.time()  # Update version after changes

    def build_tree(self, df_summary: pd.DataFrame, level: int) -> Dict:
        if level > MAX_RECURSION_LEVELS:
            return {}
        
        tree = {level: {}}
        for _, row in df_summary.iterrows():
            tree[level][row['cluster']] = {
                'summary': row['summaries'],
                'embedding': self.embd.embed_documents([row['summaries']])[0]
            }
        
        if len(df_summary) > 1:
            next_level_texts = df_summary['summaries'].tolist()
            _, next_df_summary = self.embed_cluster_summarize_texts(next_level_texts, level + 1)
            tree.update(self.build_tree(next_df_summary, level + 1))
        
        return tree

    def load_existing_knowledge(self) -> Dict:
        if VERBOSE:
            print("Loading existing knowledge from vector store")
        # Implement logic to retrieve existing embeddings and summaries from Pinecone
        # This is a placeholder and needs to be implemented based on your Pinecone setup
        return {}

    def process_new_texts(self, new_texts: List[str]) -> List[Dict]:
        if VERBOSE:
            print(f"Processing {len(new_texts)} new texts")
        embeddings = embed_texts(new_texts, self.embd)
        return [{'text': text, 'embedding': emb} for text, emb in zip(new_texts, embeddings)]

    def integrate_new_content(self, existing_data: Dict, new_data: List[Dict]) -> Dict:
        if VERBOSE:
            print("Integrating new content with existing data")
        # Implement clustering logic here
        # This is a placeholder and needs to be implemented with your clustering algorithm
        return {}

    def update_summaries(self, combined_data: Dict) -> Dict:
        if VERBOSE:
            print("Updating summaries for modified or new clusters")
        # Implement summary generation logic here
        # This is a placeholder and needs to be implemented with your summarization method
        return {}

    def propagate_updates(self, tree: Dict, level: int) -> Dict:
        if VERBOSE:
            print(f"Propagating updates to level {level}")
        if level >= MAX_RECURSION_LEVELS:
            return tree
        
        # Implement logic to update higher levels of the tree
        # This is a placeholder and needs to be implemented based on your tree structure
        return self.propagate_updates(tree, level + 1)

    def update_vector_store(self, tree: Dict, df_clusters: pd.DataFrame):
        if VERBOSE:
            print("Updating vector store with new data")
        
        texts = []
        embeddings = []
        metadata_list = []
        
        # Add original documents
        for _, row in df_clusters.iterrows():
            texts.append(row['text'])
            embeddings.append(row['embd'])
            metadata_list.append({
                'level': 0,  # Use level 0 for original documents
                'cluster_id': 'original',
                'version': self.version
            })
        
        # Add summaries
        for level, clusters in tree.items():
            for cluster_id, data in clusters.items():
                texts.append(data['summary'])
                embeddings.append(data['embedding'])
                metadata_list.append({
                    'level': level,
                    'cluster_id': cluster_id,
                    'version': self.version
                })
        
        upsert_to_pinecone(self.vectorstore, texts, embeddings, metadata_list)
        
        # Verify the update
        stats = self.vectorstore.describe_index_stats()
        if VERBOSE:
            print(f"Updated Pinecone index stats: {stats}")

    def generate_id(self, level: int, cluster_id: str, is_old: bool) -> str:
        prefix = 'old' if is_old else 'new'
        return f"{prefix}_summary_level{level}_cluster{cluster_id}_{self.version}"

    def get_version_history(self) -> List[float]:
        if VERBOSE:
            print("Retrieving version history")
        # Implement logic to retrieve version history from vector store metadata
        # This is a placeholder and needs to be implemented based on your metadata structure
        return []

    def rollback_to_version(self, target_version: float):
        if VERBOSE:
            print(f"Rolling back to version {target_version}")
        # Implement logic to rollback the tree to a specific version
        # This is a placeholder and needs to be implemented based on your versioning system
        pass

    def embed_cluster_texts(self, texts: List[str]) -> pd.DataFrame:
        """
        Embeds a list of texts and clusters them, returning a DataFrame with texts, their embeddings, and cluster labels.

        Parameters:
        - texts: List[str], a list of text documents to be processed.

        Returns:
        - pandas.DataFrame: A DataFrame containing the original texts, their embeddings, and the assigned cluster labels.
        """
        if VERBOSE:
            print(f"Embedding and clustering {len(texts)} texts")
        text_embeddings_np = embed_texts(texts, self.embd)  # Generate embeddings
        cluster_labels = self.perform_clustering(
            text_embeddings_np, CLUSTERING_DIM, CLUSTERING_THRESHOLD
        )  # Perform clustering on the embeddings
        df = pd.DataFrame()  # Initialize a DataFrame to store the results
        df["text"] = texts  # Store original texts
        df["embd"] = list(text_embeddings_np)  # Store embeddings as a list in the DataFrame
        df["cluster"] = cluster_labels  # Store cluster labels
        return df

    def perform_clustering(self, embeddings: np.ndarray, dim: int, threshold: float) -> List[np.ndarray]:
        """
        Perform clustering on the embeddings by first reducing their dimensionality globally, then clustering
        using a Gaussian Mixture Model, and finally performing local clustering within each global cluster.

        Parameters:
        - embeddings: The input embeddings as a numpy array.
        - dim: The target dimensionality for UMAP reduction.
        - threshold: The probability threshold for assigning an embedding to a cluster in GMM.

        Returns:
        - A list of numpy arrays, where each array contains the cluster IDs for each embedding.
        """
        if len(embeddings) <= dim + 1:
            # Avoid clustering when there's insufficient data
            return [np.array([0]) for _ in range(len(embeddings))]

        # Global dimensionality reduction
        reduced_embeddings_global = self.global_cluster_embeddings(embeddings, dim)
        # Global clustering
        global_clusters, n_global_clusters = self.GMM_cluster(reduced_embeddings_global, threshold)

        all_local_clusters = [np.array([]) for _ in range(len(embeddings))]
        total_clusters = 0

        # Iterate through each global cluster to perform local clustering
        for i in range(n_global_clusters):
            # Extract embeddings belonging to the current global cluster
            global_cluster_embeddings_ = embeddings[np.array([i in gc for gc in global_clusters])]

            if len(global_cluster_embeddings_) == 0:
                continue
            if len(global_cluster_embeddings_) <= dim + 1:
                # Handle small clusters with direct assignment
                local_clusters = [np.array([0]) for _ in global_cluster_embeddings_]
                n_local_clusters = 1
            else:
                # Local dimensionality reduction and clustering
                reduced_embeddings_local = self.local_cluster_embeddings(global_cluster_embeddings_, dim)
                local_clusters, n_local_clusters = self.GMM_cluster(reduced_embeddings_local, threshold)

            # Assign local cluster IDs, adjusting for total clusters already processed
            for j in range(n_local_clusters):
                local_cluster_embeddings_ = global_cluster_embeddings_[np.array([j in lc for lc in local_clusters])]
                indices = np.where((embeddings == local_cluster_embeddings_[:, None]).all(-1))[1]
                for idx in indices:
                    all_local_clusters[idx] = np.append(all_local_clusters[idx], j + total_clusters)

            total_clusters += n_local_clusters

        return all_local_clusters

    def global_cluster_embeddings(self, embeddings: np.ndarray, dim: int, n_neighbors: Optional[int] = None, metric: str = "cosine") -> np.ndarray:
        """
        Perform global dimensionality reduction on the embeddings using UMAP.

        Parameters:
        - embeddings: The input embeddings as a numpy array.
        - dim: The target dimensionality for the reduced space.
        - n_neighbors: Optional; the number of neighbors to consider for each point.
                       If not provided, it defaults to the square root of the number of embeddings.
        - metric: The distance metric to use for UMAP.

        Returns:
        - A numpy array of the embeddings reduced to the specified dimensionality.
        """
        if n_neighbors is None:
            n_neighbors = int((len(embeddings) - 1) ** 0.5)
        return umap.UMAP(n_neighbors=n_neighbors, n_components=dim, metric=metric).fit_transform(embeddings)

    def local_cluster_embeddings(self, embeddings: np.ndarray, dim: int, num_neighbors: int = 10, metric: str = "cosine") -> np.ndarray:
        """
        Perform local dimensionality reduction on the embeddings using UMAP, typically after global clustering.

        Parameters:
        - embeddings: The input embeddings as a numpy array.
        - dim: The target dimensionality for the reduced space.
        - num_neighbors: The number of neighbors to consider for each point.
        - metric: The distance metric to use for UMAP.

        Returns:
        - A numpy array of the embeddings reduced to the specified dimensionality.
        """
        return umap.UMAP(n_neighbors=num_neighbors, n_components=dim, metric=metric).fit_transform(embeddings)

    def GMM_cluster(self, embeddings: np.ndarray, threshold: float, random_state: int = 0):
        """
        Cluster embeddings using a Gaussian Mixture Model (GMM) based on a probability threshold.

        Parameters:
        - embeddings: The input embeddings as a numpy array.
        - threshold: The probability threshold for assigning an embedding to a cluster.
        - random_state: Seed for reproducibility.

        Returns:
        - A tuple containing the cluster labels and the number of clusters determined.
        """
        n_clusters = self.get_optimal_clusters(embeddings)
        gm = GaussianMixture(n_components=n_clusters, random_state=random_state)
        gm.fit(embeddings)
        probs = gm.predict_proba(embeddings)
        labels = [np.where(prob > threshold)[0] for prob in probs]
        return labels, n_clusters

    def get_optimal_clusters(self, embeddings: np.ndarray, max_clusters: int = 50, random_state: int = 0) -> int:
        """
        Determine the optimal number of clusters using the Bayesian Information Criterion (BIC) with a Gaussian Mixture Model.

        Parameters:
        - embeddings: The input embeddings as a numpy array.
        - max_clusters: The maximum number of clusters to consider.
        - random_state: Seed for reproducibility.

        Returns:
        - An integer representing the optimal number of clusters found.
        """
        max_clusters = min(max_clusters, len(embeddings))
        n_clusters = np.arange(1, max_clusters)
        bics = []
        for n in n_clusters:
            gm = GaussianMixture(n_components=n, random_state=random_state)
            gm.fit(embeddings)
            bics.append(gm.bic(embeddings))
        return n_clusters[np.argmin(bics)]

    def fmt_txt(self, df: pd.DataFrame) -> str:
        """
        Format the text documents in a DataFrame into a single string.

        Args:
        df (pd.DataFrame): DataFrame containing the 'text' column with text documents to format.

        Returns:
        str: A single string where all text documents are joined by a specific delimiter.
        """
        unique_txt = df["text"].tolist()
        return "--- --- \n --- --- ".join(unique_txt)

    def embed_cluster_summarize_texts(self, texts: List[str], level: int) -> Tuple[pd.DataFrame, pd.DataFrame]:
        if VERBOSE:
            print(f"Starting embed_cluster_summarize_texts for level {level}")
            print(f"Number of texts to process: {len(texts)}")

        df_clusters = self.embed_cluster_texts(texts)

        expanded_list = []
        for index, row in df_clusters.iterrows():
            for cluster in row["cluster"]:
                expanded_list.append(
                    {"text": row["text"], "embd": row["embd"], "cluster": cluster}
                )

        expanded_df = pd.DataFrame(expanded_list)
        all_clusters = expanded_df["cluster"].unique()

        if VERBOSE:
            print(f"--Generated {len(all_clusters)} clusters--")

        prompt = ChatPromptTemplate.from_template(SUMMARIZATION_PROMPT)
        chain = prompt | self.model | StrOutputParser()

        summaries = []
        for i in all_clusters:
            if VERBOSE:
                print(f"Summarizing cluster {i}")
            df_cluster = expanded_df[expanded_df["cluster"] == i]
            formatted_txt = self.fmt_txt(df_cluster)
            summaries.append(chain.invoke({"text": formatted_txt}))

        df_summary = pd.DataFrame(
            {
                "summaries": summaries,
                "level": [level] * len(summaries),
                "cluster": list(all_clusters),
            }
        )

        if VERBOSE:
            print(f"Completed embed_cluster_summarize_texts for level {level}")
            print(f"Number of summaries generated: {len(summaries)}")

        return df_clusters, df_summary

    def fmt_txt(self, df: pd.DataFrame) -> str:
        """
        Format the text documents in a DataFrame into a single string.

        Args:
        df (pd.DataFrame): DataFrame containing the 'text' column with text documents to format.

        Returns:
        str: A single string where all text documents are joined by a specific delimiter.
        """
        unique_txt = df["text"].tolist()
        return "--- --- \n --- --- ".join(unique_txt)

# You may need to implement additional helper functions or import them from other modules
