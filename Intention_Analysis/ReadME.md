# Analysis of our Intent Annotations
After each annotator (six in total) annotated the intent of 35 prompts, we performed an analysis to determine inter-annotator agreement for both the intent and the perceived harmfulness of the prompt.

In addition to this, we also got a baseline by retrieving the zero-shot intent annotations produced by an LLM (`RedHatAI/Qwen3-8B-quantized.w4a16` from Huggingface), and we compared inter-annotator agreement within the human annotators compared to the LLM's annotations. The procedure used to obtain the LLM's intent annotations can be found in `vllm_baseline.py` (Note that a non-vllm version can be found in `LLM_baseline.py`). The guidelines given to the model in the prompt can be found in `Annotations/annotation_guidelines_prompt.txt`. The requirements to run this file can be found in `vllm_requirements.txt`. 

Results of the group annotations and LLM-generated annotations can be found in the `Annotations` folder. Note that three LLMs were tested for the baseline, but the `Qwen3-8B-quantized.w4a16` was used as the final model as it is the largest model, hence it is expected to work the best.

The annotation analyses and their results can be found in `similarity_analysis.ipynb`. 