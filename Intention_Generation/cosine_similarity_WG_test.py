import numpy as np
import pandas as pd
import os

from cosine_similarity_base import *

def read_data(file_path):
    full_file_path = "predictions/test/" + file_path
    test_data_intents = pd.read_json(full_file_path, lines=True)
    pred_intents = test_data_intents['prediction']
    true_intents = test_data_intents['true_intent']

    return pred_intents, true_intents

def main():
    model = get_embedding_model()

    similarity_results = []
    file_paths = os.listdir("predictions/test")
    for file_path in file_paths:
        pred_intents, true_intents = read_data(file_path)

        emb_pred = model.encode(pred_intents.tolist())
        emb_true = model.encode(true_intents.tolist())

        similarity = get_index_wisecosine_similarity(emb_pred, emb_true)
        mean_similarity = np.mean(similarity)

        similarity_results.append({
        "Model": file_path[:-10],
        "Cosine Similarity (Model Prediction vs Human Label)": round(mean_similarity, 3),
        })

    sim_results_df = pd.DataFrame(similarity_results)
    with open("analysis/t5_full_similarity_summary_wild_gaurd_test.csv", "w", encoding="utf-8") as f:
        f.write("\n")
        sim_results_df.to_csv(f, index=False)


if __name__ == "__main__":
    main()
    
