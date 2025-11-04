# Welcome to the Harm Classification section

In this section, we train an ensemble of LLMs to predict whether a prompt is `harmful` or `safe`. The ensemble is then used to get an uncertainty quantification per prompt, to determine which prompts the model has the most difficulty classifying.

## Training the Model
The ensemble model is trained in `train_harm_model.py`. One ensemble consists of six base models. We investigate three variations of Bert as the base model: DistilBert, Bert, and Roberta. For each variant, an ensemble is trained and uncertainty is investigated.

The data used to train the models is sourced from the WildGuardMix dataset. We sample 10,000 data points from WildGuardMix, which are then randomly split into a 70-15-15% train-validation-test split.

Each base model in the ensemble is trained on the train split, and performance during training is monitored on the validation split. After training is complete, the final performance of the model of the model is evaluated on the test split.

Models are trained for 3 epochs with a batch size of 16 and a learning rate of 1e-5.

After training, the final model predictions of the ensemble are taken as the class with the highest mean logits across the base models. 

The final test F1 scores for each variant of the ensemple were:

| Model      | Test F1 Score |
| ----------- | ----------- |
| DistilBert      |   0.78     |
| Bert   | 0.80        |
| Roberta |    0.84    |

## Uncertainty Quantification
Our implementationof Uncertainty Quantification for our ensembles can be found in `uncertainty.py`. The uncertainty of an ensemble of classifiers can be determined by getting the entropy of the mean logits across the base models. That is, the logits per class (`harmful` and `safe`) are averaged, and the entropy of the logits is determined from their softmaxed proabilities. This gives a measure of model confidence, with higher entropy indicating higher uncertainty (lower confidence).

## Analysis
For each ensemble variant (DistilBert, Bert, and Roberta), we investigate the uncertainty of the model, the most difficult prompts to categorise according to the uncertainty, and other trends in uncertainty. These can all be found in `harm_classification_analysis.ipynb`, please refer to that file for the analyses.
