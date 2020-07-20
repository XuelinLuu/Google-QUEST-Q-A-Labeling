import torch
class BERTDatasetTraining:
    def __init__(self,
                 q_title,
                 q_body,
                 answer,
                 targets,
                 tokenizer,
                 max_len
                 ):
        self.q_title = q_title
        self.q_body = q_body
        self.answer = answer
        self.targets = targets
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.answer)

    def __getitem__(self, item):
        question_title = str(self.q_title[item])
        question_body = str(self.q_body[item])
        answer = str(self.answer[item])
        targets = self.targets[item, :]
        inputs = self.tokenizer(
            question_title + " " + question_body,
            answer,
            add_special_tokens=True,
        )

        ids = inputs['input_ids']
        attention_mask = inputs["attention_mask"]
        token_type_id = inputs["token_type_ids"]

        if len(ids) > self.max_len:
            ids = ids[:self.max_len]
            attention_mask = attention_mask[:self.max_len]
            token_type_id = token_type_id[:self.max_len]
        elif len(ids) < self.max_len:

            padding_len = self.max_len - len(ids)
            ids = ids + ([0] * padding_len)
            attention_mask = attention_mask + ([0] * padding_len)
            token_type_id = token_type_id + ([0] * padding_len)

        return {
            "ids": torch.tensor(ids, dtype=torch.long),
            "token_type_ids": torch.tensor(token_type_id, dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
            "targets": torch.tensor(targets, dtype=torch.float)
        }