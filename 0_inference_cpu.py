import torch
from torch import nn

import torch
from torch import nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import gluonnlp as nlp
import numpy as np
from tqdm import tqdm
from tqdm.notebook import tqdm, tqdm_notebook
from datetime import datetime

from kobert.utils import get_tokenizer
from kobert.pytorch_kobert import get_pytorch_kobert_model

from transformers import AdamW
from transformers.optimization import get_cosine_schedule_with_warmup

##GPU 사용 시
device = torch.device("cpu")

bertmodel, vocab = get_pytorch_kobert_model()

# wget https://www.dropbox.com/s/374ftkec978br3d/ratings_train.txt?dl=1
# wget https://www.dropbox.com/s/977gbwh542gdy94/ratings_test.txt?dl=1

dataset_train = nlp.data.TSVDataset("ratings_train.txt?dl=1", field_indices=[1,2], num_discard_samples=1)
dataset_test = nlp.data.TSVDataset("ratings_test.txt?dl=1", field_indices=[1,2], num_discard_samples=1)
# print(dataset_test2[0])
# ['굳 ㅋ', '1']
# dataset_test=dataset_test2[0]

tokenizer = get_tokenizer()
tok = nlp.data.BERTSPTokenizer(tokenizer, vocab, lower=False)

class BERTDataset(Dataset):
    def __init__(self, dataset, sent_idx, label_idx, bert_tokenizer, max_len,
                 pad, pair):
        transform = nlp.data.BERTSentenceTransform(
            bert_tokenizer, max_seq_length=max_len, pad=pad, pair=pair)

        self.sentences = [transform([i[sent_idx]]) for i in dataset]
        self.labels = [np.int32(i[label_idx]) for i in dataset]

    def __getitem__(self, i):
        return (self.sentences[i] + (self.labels[i], ))

    def __len__(self):
        return (len(self.labels))


## Setting parameters
max_len = 64
batch_size = 64
warmup_ratio = 0.1
num_epochs = 5
max_grad_norm = 1
log_interval = 200
learning_rate =  5e-5

data_train = BERTDataset(dataset_train, 0, 1, tok, max_len, True, False)
data_test = BERTDataset(dataset_test, 0, 1, tok, max_len, True, False)

train_dataloader = torch.utils.data.DataLoader(data_train, batch_size=batch_size, num_workers=5)
test_dataloader = torch.utils.data.DataLoader(data_test, batch_size=batch_size, num_workers=5)


class BERTClassifier(nn.Module):
    def __init__(self,
                 bert,
                 hidden_size = 768,
                 num_classes=2,
                 dr_rate=None,
                 params=None):
        super(BERTClassifier, self).__init__()
        self.bert = bert
        self.dr_rate = dr_rate
                 
        self.classifier = nn.Linear(hidden_size , num_classes)
        if dr_rate:
            self.dropout = nn.Dropout(p=dr_rate)
    
    def gen_attention_mask(self, token_ids, valid_length):
        attention_mask = torch.zeros_like(token_ids)
        for i, v in enumerate(valid_length):
            attention_mask[i][:v] = 1
        return attention_mask.float()

    def forward(self, token_ids, valid_length, segment_ids):
        attention_mask = self.gen_attention_mask(token_ids, valid_length)
        
        _, pooler = self.bert(input_ids = token_ids, token_type_ids = segment_ids.long(), attention_mask = attention_mask.float().to(token_ids.device))
        if self.dr_rate:
            out = self.dropout(pooler)
        return self.classifier(out)



#GPU에서 저장하고 CPU에서 불러오기
PATH = 'model_5epoch.pt'

# # 불러오기
model = BERTClassifier(bertmodel,  dr_rate=0.5)
model.load_state_dict(torch.load(PATH, map_location=device))

# Prepare optimizer and schedule (linear warmup and decay)
no_decay = ['bias', 'LayerNorm.weight']
optimizer_grouped_parameters = [
    {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
    {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
]


optimizer = AdamW(optimizer_grouped_parameters, lr=learning_rate)
loss_fn = nn.CrossEntropyLoss()

t_total = len(train_dataloader) * num_epochs
warmup_step = int(t_total * warmup_ratio)

scheduler = get_cosine_schedule_with_warmup(optimizer, num_warmup_steps=warmup_step, num_training_steps=t_total)


def calc_accuracy(X,Y):
    max_vals, max_indices = torch.max(X, 1)
    train_acc = (max_indices == Y).sum().data.cpu().numpy()/max_indices.size()[0]
    return train_acc

def timedelta_total_seconds(timedelta):
    return (timedelta.microseconds + 0.0 + (timedelta.seconds + timedelta.days * 24 * 3600) * 10 ** 6) / 10 ** 6    

model.eval()

test_acc = 0.0
test_num = 0
test_s = 0
for batch_id, (token_ids, valid_length, segment_ids, label) in enumerate(tqdm_notebook(test_dataloader)):
    t_start = datetime.now()
    token_ids = token_ids.long().to(device)
    segment_ids = segment_ids.long().to(device)
    valid_length= valid_length
    label = label.long().to(device)
    out = model(token_ids, valid_length, segment_ids)
    test_acc += calc_accuracy(out, label)
    t_end = datetime.now()
    test_num += 1
    test_s += timedelta_total_seconds(t_end - t_start)
    print("inference 시간 : ", t_end - t_start)
print("test acc {}".format( test_acc / (batch_id+1)))
print('test 개수 : ',test_num)
print('test 전체 시간 : ',test_s)
print('평균 시간 : ', test_s/test_num)

'''
inference 시간 :  0:00:01.428559
inference 시간 :  0:00:01.461723
inference 시간 :  0:00:01.759260
inference 시간 :  0:00:01.825048
inference 시간 :  0:00:01.396874

...

inference 시간 :  0:00:01.277479
inference 시간 :  0:00:01.734455
inference 시간 :  0:00:01.694438
inference 시간 :  0:00:01.609790
inference 시간 :  0:00:00.456919
'''

'''
test acc 0.8979379795396419
test 개수 :  782
test 전체 시간 :  1216.5802919999992
평균 시간 :  1.5557292736572879
'''
