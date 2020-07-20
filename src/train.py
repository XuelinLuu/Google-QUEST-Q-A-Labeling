import torch
import transformers
import pandas as pd
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader
from sklearn import model_selection

from dataset import *
from model import *
from engine import *



def run():
    MAX_LEN = 512
    TRAIN_BATCH_SIZE = 4
    EPOCHS = 10

    # prepare for data
    df_data = pd.read_csv("../input/train.csv").fillna("none")
    train_data, valid_data = model_selection.train_test_split(df_data, random_state=4, test_size=0.1)
    train_data = train_data.reset_index(drop=True)
    valid_data = valid_data.reset_index(drop=True)

    sample = pd.read_csv("../input/sample_submission.csv")
    sample_columns = list(sample.drop("qa_id", axis=1).columns)

    train_targets = train_data[sample_columns].values
    valid_targets = valid_data[sample_columns].values

    tokenizer = transformers.BertTokenizer.from_pretrained("bert-base-uncased")


    train_dataset = BERTDatasetTraining(
        q_title=train_data.question_title.values,
        q_body=train_data.question_body.values,
        answer=train_data.answer.values,
        targets=train_targets,
        tokenizer=tokenizer,
        max_len=MAX_LEN
    )
    train_dataloader = DataLoader(
        dataset=train_dataset,
        batch_size=TRAIN_BATCH_SIZE,
        shuffle=True
    )

    valid_dataset = BERTDatasetTraining(
        q_title=valid_data.question_title.values,
        q_body=valid_data.question_body.values,
        answer=valid_data.answer.values,
        targets=valid_targets,
        tokenizer=tokenizer,
        max_len=MAX_LEN
    )
    valid_dataloader = DataLoader(
        dataset=valid_dataset,
        batch_size=4,
        shuffle=False
    )

    lr = 1e-3
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = BERTBaseUncased("bert-base-uncased").to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.7)

    for epoch in range(EPOCHS):
        print(f"---------------------------{epoch}-------------------------------------")
        train_fn(model=model, device=device, data_loader=train_dataloader, optimizer=optimizer, scheduler=scheduler)
        eval_fn(model=model, device=device, data_loader=valid_dataloader)
        model_output = f"../models/epochs_{epoch}.pkl"
        torch.save(model.states_dict(), model_output)

if __name__ == '__main__':
    run()