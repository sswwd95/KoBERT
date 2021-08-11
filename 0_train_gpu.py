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
device = torch.device("cuda:0")

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
# max_len = 64
# batch_size = 64
# warmup_ratio = 0.1
# num_epochs = 5
# max_grad_norm = 1
# log_interval = 200
# learning_rate =  5e-5

max_len = 64
batch_size = 64
warmup_ratio = 0.1
num_epochs = 1
max_grad_norm = 1
log_interval = 200
learning_rate =  1e-1

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


model = BERTClassifier(bertmodel,  dr_rate=0.5).to(device)

# PATH = 'model_5epoch.pt'
PATH = 'test.pt'

# # 저장하기
torch.save(model.state_dict(), PATH)

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


# train 

start = datetime.now()
for e in range(num_epochs):
    train_acc = 0.0
    test_acc = 0.0
    model.train()
    tr_start = datetime.now()
    for batch_id, (token_ids, valid_length, segment_ids, label) in enumerate(tqdm_notebook(train_dataloader)):
        optimizer.zero_grad()
        token_ids = token_ids.long().to(device)
        segment_ids = segment_ids.long().to(device)
        valid_length= valid_length
        label = label.long().to(device)
        out = model(token_ids, valid_length, segment_ids)
        loss = loss_fn(out, label)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
        optimizer.step()
        scheduler.step()  # Update learning rate schedule
        train_acc += calc_accuracy(out, label)
        if batch_id % log_interval == 0:
            print("epoch {} batch id {} loss {} train acc {}".format(e+1, batch_id+1, loss.data.cpu().numpy(), train_acc / (batch_id+1)))
    tr_end = datetime.now()

    print("train 시간 : ", tr_end - tr_start)
    print("epoch {} train acc {}".format(e+1, train_acc / (batch_id+1)))

end = datetime.now()

print(" 총 train 시간 : ", end - start)

PATH = 'test.pt'

# # 저장하기
torch.save(model.state_dict(), PATH)

print('저장 완료')


'''
epoch 1 batch id 1 loss 0.7100536227226257 train acc 0.5
epoch 1 batch id 201 loss 0.5784103870391846 train acc 0.5387904228855721
epoch 1 batch id 401 loss 0.45215731859207153 train acc 0.6500155860349127
epoch 1 batch id 601 loss 0.4518764019012451 train acc 0.7094685940099834
epoch 1 batch id 801 loss 0.3956311047077179 train acc 0.7424118289637952
epoch 1 batch id 1001 loss 0.28905555605888367 train acc 0.7637518731268731
epoch 1 batch id 1201 loss 0.2709890305995941 train acc 0.7792334512905912
epoch 1 batch id 1401 loss 0.28721883893013 train acc 0.7898599214846538
epoch 1 batch id 1601 loss 0.3790733814239502 train acc 0.7992856027482823
epoch 1 batch id 1801 loss 0.24800945818424225 train acc 0.8069562048861744
epoch 1 batch id 2001 loss 0.2418363094329834 train acc 0.8139367816091954
epoch 1 batch id 2201 loss 0.318508118391037 train acc 0.8196913334847796
train 시간 :  0:10:52.580520
epoch 1 train acc 0.8234788289249146
  0%|          | 0/2344 [00:00<?, ?it/s]
The current process just got forked. Disabling parallelism to avoid deadlocks...
To disable this warning, please explicitly set TOKENIZERS_PARALLELISM=(true | false)
The current process just got forked. Disabling parallelism to avoid deadlocks...
To disable this warning, please explicitly set TOKENIZERS_PARALLELISM=(true | false)
The current process just got forked. Disabling parallelism to avoid deadlocks...
To disable this warning, please explicitly set TOKENIZERS_PARALLELISM=(true | false)
The current process just got forked. Disabling parallelism to avoid deadlocks...
To disable this warning, please explicitly set TOKENIZERS_PARALLELISM=(true | false)
The current process just got forked. Disabling parallelism to avoid deadlocks...
To disable this warning, please explicitly set TOKENIZERS_PARALLELISM=(true | false)
epoch 2 batch id 1 loss 0.5069687366485596 train acc 0.796875
epoch 2 batch id 201 loss 0.21414341032505035 train acc 0.8813743781094527
epoch 2 batch id 401 loss 0.23916959762573242 train acc 0.8830657730673317
epoch 2 batch id 601 loss 0.3974239230155945 train acc 0.8869332362728786
epoch 2 batch id 801 loss 0.2646070718765259 train acc 0.8895521223470662
epoch 2 batch id 1001 loss 0.28880029916763306 train acc 0.8916396103896104
epoch 2 batch id 1201 loss 0.29631900787353516 train acc 0.8942287676935887
epoch 2 batch id 1401 loss 0.18609510362148285 train acc 0.8966140256959315
epoch 2 batch id 1601 loss 0.2355034500360489 train acc 0.8985497345409119
epoch 2 batch id 1801 loss 0.16854354739189148 train acc 0.9006107717934481
epoch 2 batch id 2001 loss 0.21157340705394745 train acc 0.9024081709145427
epoch 2 batch id 2201 loss 0.2501266598701477 train acc 0.9040421967287596
train 시간 :  0:10:52.071623
epoch 2 train acc 0.905325654152446
  0%|          | 0/2344 [00:00<?, ?it/s]
The current process just got forked. Disabling parallelism to avoid deadlocks...
To disable this warning, please explicitly set TOKENIZERS_PARALLELISM=(true | false)
The current process just got forked. Disabling parallelism to avoid deadlocks...
To disable this warning, please explicitly set TOKENIZERS_PARALLELISM=(true | false)
The current process just got forked. Disabling parallelism to avoid deadlocks...
To disable this warning, please explicitly set TOKENIZERS_PARALLELISM=(true | false)
The current process just got forked. Disabling parallelism to avoid deadlocks...
To disable this warning, please explicitly set TOKENIZERS_PARALLELISM=(true | false)
The current process just got forked. Disabling parallelism to avoid deadlocks...
To disable this warning, please explicitly set TOKENIZERS_PARALLELISM=(true | false)
epoch 3 batch id 1 loss 0.42780378460884094 train acc 0.859375
epoch 3 batch id 201 loss 0.10699654370546341 train acc 0.921875
epoch 3 batch id 401 loss 0.21343857049942017 train acc 0.9243298004987531
epoch 3 batch id 601 loss 0.24652355909347534 train acc 0.9266846921797005
epoch 3 batch id 801 loss 0.22976484894752502 train acc 0.9295997191011236
epoch 3 batch id 1001 loss 0.18670186400413513 train acc 0.9316152597402597
epoch 3 batch id 1201 loss 0.1386650651693344 train acc 0.9339092422980849
epoch 3 batch id 1401 loss 0.12032661586999893 train acc 0.9353809778729479
epoch 3 batch id 1601 loss 0.10505086928606033 train acc 0.9367094784509682
epoch 3 batch id 1801 loss 0.11793763190507889 train acc 0.9384716823986674
epoch 3 batch id 2001 loss 0.11147525161504745 train acc 0.9397176411794103
epoch 3 batch id 2201 loss 0.15474936366081238 train acc 0.9406803725579282
train 시간 :  0:10:55.005044
epoch 3 train acc 0.9418728668941979
  0%|          | 0/2344 [00:00<?, ?it/s]
The current process just got forked. Disabling parallelism to avoid deadlocks...
To disable this warning, please explicitly set TOKENIZERS_PARALLELISM=(true | false)
The current process just got forked. Disabling parallelism to avoid deadlocks...
To disable this warning, please explicitly set TOKENIZERS_PARALLELISM=(true | false)
The current process just got forked. Disabling parallelism to avoid deadlocks...
To disable this warning, please explicitly set TOKENIZERS_PARALLELISM=(true | false)
The current process just got forked. Disabling parallelism to avoid deadlocks...
To disable this warning, please explicitly set TOKENIZERS_PARALLELISM=(true | false)
The current process just got forked. Disabling parallelism to avoid deadlocks...
To disable this warning, please explicitly set TOKENIZERS_PARALLELISM=(true | false)
epoch 4 batch id 1 loss 0.4557704031467438 train acc 0.84375
epoch 4 batch id 201 loss 0.08144645392894745 train acc 0.9532804726368159
epoch 4 batch id 401 loss 0.12588565051555634 train acc 0.9547615336658354
epoch 4 batch id 601 loss 0.11692456156015396 train acc 0.9564787853577371
epoch 4 batch id 801 loss 0.10580773651599884 train acc 0.9589575530586767
epoch 4 batch id 1001 loss 0.043045151978731155 train acc 0.9601648351648352
epoch 4 batch id 1201 loss 0.17575861513614655 train acc 0.9616725645295587
epoch 4 batch id 1401 loss 0.030962655320763588 train acc 0.9625713775874375
epoch 4 batch id 1601 loss 0.07804004102945328 train acc 0.9632749063085572
epoch 4 batch id 1801 loss 0.06506724655628204 train acc 0.9644555108273182
epoch 4 batch id 2001 loss 0.02816619910299778 train acc 0.9652205147426287
epoch 4 batch id 2201 loss 0.12037896364927292 train acc 0.9658748864152658
train 시간 :  0:10:55.528754
epoch 4 train acc 0.9663435900170648
  0%|          | 0/2344 [00:00<?, ?it/s]
The current process just got forked. Disabling parallelism to avoid deadlocks...
To disable this warning, please explicitly set TOKENIZERS_PARALLELISM=(true | false)
The current process just got forked. Disabling parallelism to avoid deadlocks...
To disable this warning, please explicitly set TOKENIZERS_PARALLELISM=(true | false)
The current process just got forked. Disabling parallelism to avoid deadlocks...
To disable this warning, please explicitly set TOKENIZERS_PARALLELISM=(true | false)
The current process just got forked. Disabling parallelism to avoid deadlocks...
To disable this warning, please explicitly set TOKENIZERS_PARALLELISM=(true | false)
The current process just got forked. Disabling parallelism to avoid deadlocks...
To disable this warning, please explicitly set TOKENIZERS_PARALLELISM=(true | false)
epoch 5 batch id 1 loss 0.34802085161209106 train acc 0.90625
epoch 5 batch id 201 loss 0.026554860174655914 train acc 0.9738028606965174
epoch 5 batch id 401 loss 0.03854559361934662 train acc 0.9756468204488778
epoch 5 batch id 601 loss 0.12049806863069534 train acc 0.9763415141430949
epoch 5 batch id 801 loss 0.008661935105919838 train acc 0.9767283083645443
epoch 5 batch id 1001 loss 0.02365855500102043 train acc 0.9771634615384616
epoch 5 batch id 1201 loss 0.042416561394929886 train acc 0.9775967943380516
epoch 5 batch id 1401 loss 0.017520088702440262 train acc 0.9779287116345468
epoch 5 batch id 1601 loss 0.08207011222839355 train acc 0.9783436133666459
epoch 5 batch id 1801 loss 0.013480879366397858 train acc 0.9788398806218768
epoch 5 batch id 2001 loss 0.029002776369452477 train acc 0.9788230884557722
epoch 5 batch id 2201 loss 0.07336822897195816 train acc 0.9789655270331667
train 시간 :  0:10:53.277821
epoch 5 train acc 0.9791955524744027
'''