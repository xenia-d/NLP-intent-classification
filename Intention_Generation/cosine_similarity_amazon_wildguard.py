from sentence_transformers import SentenceTransformer
import numpy as np
from scipy.stats import pearsonr
import pandas as pd
import matplotlib.pyplot as plt

from cosine_similarity_base import *


if __name__ == "__main__":
    model = get_embedding_model()

    humans = pd.read_csv("our_amazon_science_annotations.csv")
    humans.columns = ["prompt", "ann1", "ann2", "ann3"]

    emb_ann1 = model.encode(humans["ann1"].tolist())
    emb_ann2 = model.encode(humans["ann2"].tolist())
    emb_ann3 = model.encode(humans["ann3"].tolist())

    # Compute inter-annotator agreement
    human_agreement_matrix = get_cosine_similarity_agreement_between_individuals([emb_ann1, emb_ann2, emb_ann3])
    human_agreement_df = pd.DataFrame(
        human_agreement_matrix,
        index=["Ann1", "Ann2", "Ann3"],
        columns=["Ann1", "Ann2", "Ann3"]
    ).round(3)

    overall_mean_agreement = np.mean([human_agreement_matrix[i][j] for i in range(len(human_agreement_matrix)) for j in range(len(human_agreement_matrix)) if i != j])

    human_agreement_df.to_csv("analysis/human_interannotator_agreement.csv", index=True)
    print("Saved: analysis/human_interannotator_agreement.csv")
    print(human_agreement_df)
    print(f"Overall Mean Inter-Annotator Agreement: {overall_mean_agreement:.3f}")


    # Compute **pairwise annotator agreement per prompt**
    sim_12 = np.sum(emb_ann1 * emb_ann2, axis=1)
    sim_13 = np.sum(emb_ann1 * emb_ann3, axis=1)
    sim_23 = np.sum(emb_ann2 * emb_ann3, axis=1)

    mean_human_agreement = (sim_12 + sim_13 + sim_23) / 3.0
    humans["mean_human_agreement"] = mean_human_agreement

    model_files = {
        "T5-small": "generated_intents/t5-small_AmazonScience_intents.json",
        "T5-base": "generated_intents/t5-base_AmazonScience_intents.json",
        "T5-large": "generated_intents/t5-large_AmazonScience_intents.json"
    }

    summary = pd.DataFrame({"prompt": humans["prompt"]})
    summary_results = []
    correlation_results = []

    for model_name, file in model_files.items():
        print(f"\n=== Processing {model_name} ===")

        llm = pd.read_json(file)
        df = humans.merge(llm, on="prompt", how="inner")

        emb_llm = model.encode(df["generated_intent"].tolist())
        emb_ann1 = model.encode(df["ann1"].tolist())
        emb_ann2 = model.encode(df["ann2"].tolist())
        emb_ann3 = model.encode(df["ann3"].tolist())

        sim1 = get_index_wisecosine_similarity(emb_llm, emb_ann1)
        sim2 = get_index_wisecosine_similarity(emb_llm, emb_ann2)
        sim3 = get_index_wisecosine_similarity(emb_llm, emb_ann3)

        mean_llm_sim = np.mean([sim1, sim2, sim3], axis=0)
        df["mean_llm_similarity"] = mean_llm_sim

        summary = summary.merge(df[["prompt"]], on="prompt", how="right")
        summary[f"{model_name} vs Ann1"] = sim1
        summary[f"{model_name} vs Ann2"] = sim2
        summary[f"{model_name} vs Ann3"] = sim3

        # Compute mean per annotator + mean across all annotators
        summary_results.append({
            "Model": model_name,
            "T5 vs Ann1": round(np.mean(sim1), 3),
            "T5 vs Ann2": round(np.mean(sim2), 3),
            "T5 vs Ann3": round(np.mean(sim3), 3),
            "T5 vs All Ann": round(np.mean([np.mean(sim1), np.mean(sim2), np.mean(sim3)]), 3)
        })

        corr, pval = pearsonr(df["mean_llm_similarity"], df["mean_human_agreement"])
        correlation_results.append({
        "Model": model_name,
        "Correlation (LLM vs Human Agreement)": round(corr, 3),
        "p-value": round(pval, 4)
        })


            # Create scatter plot
        plt.figure()
        plt.scatter(mean_llm_sim, df["mean_human_agreement"])
        plt.xlabel("LLM Similarity to Annotators (mean)")
        plt.ylabel("Human Inter-Annotator Agreement (mean)")
        plt.title(f"{model_name}: LLM-Human Agreement Correlation")

        # Save plot
        plot_path = f"analysis/{model_name}_correlation_plot.png"
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()

        print(f"Saved: {plot_path}")

    mean_summary_df = pd.DataFrame(summary_results)

    with open("analysis/t5_full_similarity_summary.csv", "w", encoding="utf-8") as f:
        f.write("\n")
        mean_summary_df.to_csv(f, index=False)


    print("Saved: analysis/t5_full_similarity_summary.csv")
    print(mean_summary_df)

    # Save correlation results
    correlation_results_df = pd.DataFrame(correlation_results)
    with open("analysis/t5_full_correlation_summary.csv", "w", encoding="utf-8") as f:
        f.write("\n")
        correlation_results_df.to_csv(f, index=False)

    print("Saved: analysis/t5_full_correlation_summary.csv")
    print(correlation_results_df)
