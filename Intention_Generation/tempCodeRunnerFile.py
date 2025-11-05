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