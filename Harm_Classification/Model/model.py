from transformers import DistilBertModel, BertModel, RobertaModel, DistilBertTokenizer, BertTokenizer, RobertaTokenizer
import torch 
import torch.nn as nn


class Model(nn.Module):
    def __init__(self, base_model, hidden_size):
        super().__init__()
        self.bert = base_model
        
        # 2 classes: harmful, unharmful
        self.classify_head = nn.Linear(in_features=hidden_size, out_features=2)

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        
        # Use [CLS] token representation for classification
        pooled_output = outputs.last_hidden_state[:, 0]
        
        # Classification head
        logits_classify = self.classify_head(pooled_output)
        
        return logits_classify
    
def get_model(device, model_name="DistilBert"):
    # Load base transformer model
    if model_name == "DistilBert":
        bert_model = DistilBertModel.from_pretrained("distilbert-base-uncased")
    elif model_name == "Bert":
        bert_model = BertModel.from_pretrained("bert-base-uncased")
    elif model_name == "Roberta":
        bert_model = RobertaModel.from_pretrained("roberta-base")

    hidden_size = bert_model.config.hidden_size
    print("Hidden Size:", hidden_size)

    # Initialize custom model with classification head
    main_model = Model(bert_model, hidden_size)

    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    main_model.to(device)

    return main_model

def get_tokenizer(model_name = "DistilBert"):
    if model_name == "DistilBert":
        tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")
    elif model_name == "Bert":
        tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    elif model_name == "Roberta":
        tokenizer = RobertaTokenizer.from_pretrained("roberta-base")
    
    return tokenizer

