from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import pandas as pd

def get_index_wisecosine_similarity(embeddings1, embeddings2):
    """
    Compute the cosine similarity between two sets of embeddings, at each index.
    That is, it computes the cosine similarity between embeddings1[i] and embeddings2[i] for all i.
    The two lists must be of the same length.

    Args:
        embeddings1 (list or np.array): First set of embeddings.
        embeddings2 (list or np.array): Second set of embeddings.

    Returns:
        np.array: Cosine similarity scores.
    """

    if len(embeddings1) != len(embeddings2):
        raise ValueError("The two embedding lists must be of the same length. List1 length: {}, List2 length: {}".format(len(embeddings1), len(embeddings2)))
    
    similarities = []
    for i in range(len(embeddings1)):
        similarity = cosine_similarity([embeddings1[i]], [embeddings2[i]])
        similarities.append(similarity[0][0])

    return similarities

def get_cosine_similarity_agreement_between_individuals(matrix_of_embeddings):
    """
    Compute the cosine similarity agreement between multiple individuals' embeddings.
    The input is a 2D list where each sublist contains embeddings from one individual.
    The function computes the pairwise cosine similarity between all individuals' embeddings
    at each index and returns the average similarity for each index.

    Args:
        matrix_of_embeddings (list of list or np.array): 2D list of embeddings from multiple individuals.

    Returns:
        np.array: Matrix of cosine similarity scores across individuals
    """

    num_individuals = len(matrix_of_embeddings)
    if num_individuals < 2:
        raise ValueError("At least two individuals' embeddings are required to compute agreement.")
    
    # Convert inputs to numpy arrays and validate shapes
    embeddings_list = [np.asarray(e) for e in matrix_of_embeddings]

    # Expect each element to be a 2D array of shape (num_intents, embedding_dim)
    if embeddings_list[0].ndim != 2:
        raise ValueError("Each individual's embeddings must be a 2D array with shape (num_intents, embedding_dim).")

    num_intents = embeddings_list[0].shape[0]
    embedding_dim = embeddings_list[0].shape[1]

    for idx, e in enumerate(embeddings_list):
        if e.ndim != 2:
            raise ValueError(f"Embeddings for individual {idx} must be 2D. Got {e.ndim}D.")
        if e.shape[0] != num_intents:
            raise ValueError("All individuals must have the same number of intents. Expected {}, got {}".format(num_intents, e.shape[0]))
        if e.shape[1] != embedding_dim:
            raise ValueError("All embeddings must have the same dimensionality. Expected {}, got {}".format(embedding_dim, e.shape[1]))

    # Compute n x n agreement matrix. For each pair (i, j) compute the average of
    # the cosine similarity between corresponding intent embeddings (diagonal of pairwise similarity).
    n = num_individuals
    agreement = np.zeros((n, n), dtype=float)

    for i in range(n):
        for j in range(i, n):
            # element-wise dot product across intents between i and j
            dots = np.sum(embeddings_list[i] * embeddings_list[j], axis=1)
            avg = float(np.mean(dots))
            agreement[i, j] = avg
            agreement[j, i] = avg

    return agreement


if __name__ == "__main__":
    # Load the SentenceTransformer model
    model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')

    humans = pd.read_csv("our_amazon_science_annotations.csv")
    humans.columns = ["prompt", "ann1", "ann2", "ann3"]
    
    model_files = {
    "T5-small": "t5-small_AmazonScience_intents.json",
    "T5-base": "t5-base_AmazonScience_intents.json",
    "T5-large": "t5-large_AmazonScience_intents.json"
    } 

    emb_ann1 = model.encode(humans["ann1"].tolist())
    emb_ann2 = model.encode(humans["ann2"].tolist())
    emb_ann3 = model.encode(humans["ann3"].tolist())

    summary_results = []

    for model_name, file in model_files.items():
        print(f"\n=== Processing {model_name} ===")

        llm = pd.read_json(file)
        df = humans.merge(llm, on="prompt", how="inner")

        emb_llm = model.encode(df["generated_intent"].tolist())

        sim1 = get_index_wisecosine_similarity(emb_llm, emb_ann1)
        sim2 = get_index_wisecosine_similarity(emb_llm, emb_ann2)
        sim3 = get_index_wisecosine_similarity(emb_llm, emb_ann3)

        per_prompt = pd.DataFrame({
            "prompt": df["prompt"],
            f"{model_name} vs Annotator 1": sim1,
            f"{model_name} vs Annotator 2": sim2,
            f"{model_name} vs Annotator 3": sim3
        })
        per_prompt.to_csv(f"{model_name}_vs_humans_per_prompt.csv", index=False)

        summary_results.append({
            "Model": model_name,
            "Mean vs Ann1": np.mean(sim1),
            "Mean vs Ann2": np.mean(sim2),
            "Mean vs Ann3": np.mean(sim3)
        })

    summary_df = pd.DataFrame(summary_results)
    summary_df.to_csv("t5_similarity_summary.csv", index=False)

    print("\n✅ Done. Files saved:")
    print(" - *_vs_humans_per_prompt.csv for each T5 model")
    print(" - t5_similarity_summary.csv (mean comparison)")
    print(summary_df)

