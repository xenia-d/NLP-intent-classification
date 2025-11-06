from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

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

    # Example list of sentences
    LLM_intents = [
        "Book a flight to New York",
        "What's the weather like today?",
        "Set a reminder for my meeting at 3 PM",
        "Play some relaxing music",
        "Find a nearby restaurant"
    ]

    human1_intents = [
        "I want to fly to New York",
        "What's the weather like today?",
        "Remind me about my meeting at 3 PM",
        "Play some calming music",
        "Find a restaurant nearby"
    ]

    human2_intents = [
        "Schedule a flight to NYC",
        "What's the weather like in NYC?",
        "Remind me about my meeting at 3 PM",
        "Play some soothing music",
        "Find a restaurant in New York"
    ]

    # Get cosine similarities between two lists of intents
    LLM_embeddings = model.encode(LLM_intents)
    human1_embeddings = model.encode(human1_intents)
    human2_embeddings = model.encode(human2_intents)
    
    # Get the index-wise cosine similarities
    index_wise_similarities = get_index_wisecosine_similarity(LLM_embeddings, human1_embeddings)
    print("Index-wise Cosine Similarities for each annotation, between LLM and Human 1:")
    print(index_wise_similarities)

    # Get the cosine similarity agreement between multiple individuals
    matrix_of_embeddings = [LLM_embeddings, human1_embeddings, human2_embeddings]
    agreement_similarities = get_cosine_similarity_agreement_between_individuals(matrix_of_embeddings)
    print("Cosine Similarity Agreement between LLM, Human 1, and Human 2:")
    print(agreement_similarities)