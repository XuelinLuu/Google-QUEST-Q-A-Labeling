import torch.nn as nn
import transformers

class BERTBaseUncased(nn.Module):
    def __init__(self, bert_path):
        super(BERTBaseUncased, self).__init__()
        self.bert_path = bert_path
        self.bert = transformers.BertModel.from_pretrained(self.bert_path)
        self.dense = nn.Linear(768, 30) # 我的是tiny bert，可以将128改为768
        self.dropout = nn.Dropout(0.3)

    def forward(self, ids, mask, type_token_ids):
        _, o2 = self.bert(ids, attention_mask=mask, token_type_ids=type_token_ids)
        bo = self.dropout(o2)
        return self.dense(bo)