# Training Intent Generation Models 

 blablabla <3
 mention the 3 T5 models, the 2 Qwen models



# Analysis of Intent Generation Models

Once the T5 and Qwen models were trained on our ~1800 annotations on the WildGuard dataset, we performed inference on 20 prompts from the `AmazonScience/FalseReject` dataset from HuggingFace, found here : https://huggingface.co/datasets/AmazonScience/FalseReject/viewer/default/train?views%5B%5D=train 

We used prompts from this dataset to analyze the model's behavior, and compare it to that of our own annotations on the same prompts.

The three of us first annotated the 20 prompts from the `AmazonScience/FalseReject` dataset ourselves, and measured our inter-annotator agreement using mean cosine similarity across the 20 prompts. This is shown in the table below. Our overall mean inter-annotator agreement is 0.611. 


|      | Ann1  | Ann2  | Ann3  |
|------|-------|-------|-------|
| Ann1 | 1.000 | 0.620 | 0.586 |
| Ann2 | 0.620 | 1.000 | 0.626 |
| Ann3 | 0.586 | 0.626 | 1.000 |
