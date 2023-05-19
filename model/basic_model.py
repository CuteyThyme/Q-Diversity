import torch
import torch.nn as nn
from transformers import BertConfig, BertModel


class BasicBertModel(nn.Module):
    def __init__(self, pretrained_path, num_class):
        super(BasicBertModel, self).__init__()
        self.bert_config = BertConfig.from_pretrained(pretrained_path)
        self.bert = BertModel.from_pretrained(pretrained_path)
        
        self.input_size = self.bert_config.hidden_size
        self.num_class = num_class
        self.classifier = nn.Linear(self.input_size, self.num_class)
        
        confounder_len = 2   ## [0: majority    1: minority]
        self.meta_classifier = nn.Linear(self.input_size, confounder_len)


    def forward(self, input_ids=None, attention_mask=None, token_type_ids=None, inputs_embeds=None):
        if inputs_embeds == None:
            outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids, output_hidden_states=True)
        else:
            outputs = self.bert(inputs_embeds=inputs_embeds, attention_mask=attention_mask, token_type_ids=token_type_ids, output_hidden_states=True)
            
        pooled_output = outputs[1]
        logits = self.classifier(pooled_output)
        return logits, pooled_output
    
    
    def get_pseudo_prediction(self, pooled_output):
        
        logits = self.meta_classifier(pooled_output)
        _, pred = logits.topk(1, 1, True, True)
        pseudo_gid = pred.squeeze()
    
        return logits, pseudo_gid
    
    
    def get_input_embeddings(self):
        return self.bert.get_input_embeddings()
    
