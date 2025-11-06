# Intent Detection of Prompts by LLMs
Welcome to our repository investigating LLMs' ability to deduce the intent of a given prompt. Our project follows the order described below. While a high-level overview of each sectoin is described here, please refer to each folder's ReadME for further information.

## 1. Investigating baseline harm classification and model uncertainty
In this section, found in the folder `Harm_Classification`, we investigate how well an LLM can classify a prompt as safe or harmful through Supervised Fine-Tuning. Then, we apply an Uncertainty Quantification (UQ) method to determine which prompts are most difficult to classify.

Note that while this is our implementation of UQ, the final outcome of this step (finding the most uncertain prompts) was provided by Group 8, who had their own implementation.


## 2. Human Annotation
Can someone maybe write this section explaining the protocol and what we did etc.

After initial group annotation, an analysis was performed to investigate the inter-annotator agreement between all annotators. First our group annotated a few prompts all together, to analyze our agreement, and what protocols to agree on to improve standardization. Then, we discussed our protocol and observations with Group 8, and all together came up with a final protocol to collectively agree on. This protocol was put together by both Group 3 and Group 8 while discussing together, and it can be found in the `Intention_Analysis` folder, in the file `intent classification annotation protocol.pdf`. 

After agreeing on this protocol, we held a small pilot annotation experiment, where we all annotated the same 35 prompts (which took around an hour) from the WildGuard dataset, and analyzed the inter-annotator agreement. We compared our inter-annotator agreement to how much an LLM baseline (`RedHatAI/Qwen3-8B-quantized.w4a16` and `Qwen/Qwen1.5-1.8B-Chat` from Huggingface) performing zero-shot intent analysis agreed with us. This was measured with mean cosine similarity.

We found that the mean cosine similarity across the 6 of us was 0.61, and the mean cosine similarity between the Qwen models and us annotators was 0.32, for both models. We also annotated harm labels for the prompts with the following categories: Completely Harmful, Uncertain Harmful, Uncertain Safe, and Completely safe, and analysed the mean Spearman Rank Correlation between our harm labels. This correlation is visualized below, and can also be found in the file `Intention_Analysis/similarity_analysis.ipynb`


![alt text](<Intention_Analysis/correlation harm label.png>)



Once we concluded that our inter-annotator was sufficient and surpassed the LLM baseline comparison, the  the six of us continued to follow our annotation protocol to annotate ~300 prompts each from the WildGuard dataset, considering an interval of the labels with maximum confidence between 0.35 and 0.65, as determined by an ensemble implemented by group 8. This resulted in a total of ~1800 prompts annotated among the 6 of us. 


Our annotation protocol is the same protocol and guidelines we prompted LLMs to follow when testing LLM intention analysis to compare with our human intent annotations.

