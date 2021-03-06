import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import os
import argparse
from mylog import Logger
import numpy as np
from sklearn.metrics import roc_auc_score
from torch.utils.data import Dataset, DataLoader, SequentialSampler, RandomSampler
import jieba
import time
from tqdm import tqdm

# from nltk.tokenize import word_tokenize
import random


def mrr_score(y_true, y_score):
    order = np.argsort(y_score)[::-1]
    y_true = np.take(y_true, order)
    rr_score = y_true / (np.arange(len(y_true)) + 1)
    return np.sum(rr_score) / np.sum(y_true)


def dcg_score(y_true, y_score, k=10):
    order = np.argsort(y_score)[::-1]
    y_true = np.take(y_true, order[:k])
    gains = 2 ** y_true - 1
    discounts = np.log2(np.arange(len(y_true)) + 2)
    return np.sum(gains / discounts)


def ndcg_score(y_true, y_score, k=10):
    best = dcg_score(y_true, y_true, k)
    actual = dcg_score(y_true, y_score, k)
    return actual / best


def ctr_score(y_true, y_score, k=1):
    order = np.argsort(y_score)[::-1]
    y_true = np.take(y_true, order[:k])
    return np.mean(y_true)


class MultiHeadAttention(nn.Module):
    def __init__(self, num_heads, head_size, inputsize):
        super(MultiHeadAttention, self).__init__()
        self.num_heads = num_heads
        self.head_size = head_size
        self.output_dim = num_heads * head_size

        self.wq = nn.Linear(inputsize, self.output_dim, bias=False)
        self.wk = nn.Linear(inputsize, self.output_dim, bias=False)
        self.wv = nn.Linear(inputsize, self.output_dim, bias=False)

    def split_heads(self, x):
        x = x.view((-1, x.size(1), self.num_heads, self.head_size))
        x = x.permute([0, 2, 1, 3])
        return x

    def forward(self, x, mask):
        q = self.wq(x)
        k = self.wk(x)
        v = self.wv(x)

        q = self.split_heads(q)
        k = self.split_heads(k)
        v = self.split_heads(v)

        matmul_qk = torch.matmul(q, k.permute([0, 1, 3, 2]))
        scaled_attention_logits = matmul_qk / math.sqrt(q.size(-1))
        if mask is not None:
            scaled_attention_logits += mask
        attention_weight = F.softmax(scaled_attention_logits, dim=3)
        output = torch.matmul(attention_weight, v)
        output = output.permute([0, 2, 1, 3])
        output = output.contiguous().view((-1, output.size(1), self.output_dim))

        return output


class TitleLayer(nn.Module):
    def __init__(self, num_words, embeddings_matrix, args):
        super(TitleLayer, self).__init__()
        self.output_dim = args.userpre_embed_size
        self.mediallayer = args.num_heads * args.head_size
        if (embeddings_matrix is not None):
            self.embedding = nn.Embedding.from_pretrained(embeddings_matrix)
        else:
            self.embedding = nn.Embedding(num_words, args.word_embed_size)
        self.dropout1 = nn.Dropout(args.droprate)
        self.dropout2 = nn.Dropout(args.droprate)
        self.multiatt = MultiHeadAttention(args.num_heads, args.head_size, args.word_embed_size)
        self.dense1 = nn.Linear(self.output_dim, self.mediallayer)

        # self.dense2 = nn.Linear(self.mediallayer, 1)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def forward(self, newstitle, user_pre_embedding):
        x = self.embedding(newstitle)

        x = self.dropout1(x)
        mask = torch.eq(newstitle, 0).float().to(self.device)
        mask = mask.masked_fill(mask == 1, -1e9)
        mask1 = torch.unsqueeze(torch.unsqueeze(mask, 1), 1)

        # print(mask1.shape)
        selfattn_output = self.multiatt(x, mask1)

        # selfattn_output = self.dropout2(selfattn_output)
        a = self.dense1(user_pre_embedding)
        a = nn.Tanh()(a)
        attention = torch.matmul(selfattn_output, torch.unsqueeze(a, 2))
        # attention = self.dense2(attention)
        # mask2 = torch.unsqueeze(mask, 2)
        # attention += mask2
        attention_weight = F.softmax(attention, 1)
        output = torch.sum(attention_weight * selfattn_output, 1)

        return output


class Categorylayer(nn.Module):
    def __init__(self, categories, args):
        super(Categorylayer, self).__init__()
        self.embedding = nn.Embedding(categories, args.categ_embed_size)
        # self.dense = Dense(output_dim)
        self.output_dim = args.categ_embed_size

    def forward(self, inputs):
        catedembed = self.embedding(inputs)
        output = catedembed.view((-1, self.output_dim))
        return output

class Authorlayer(nn.Module):
    def __init__(self, authors, args):
        super(Authorlayer, self).__init__()
        self.embedding = nn.Embedding(authors, args.author_embed_size)
        # self.dense = Dense(output_dim)
        self.output_dim = args.author_embed_size

    def forward(self, inputs):
        authorembed = self.embedding(inputs)
        output = authorembed.view((-1, self.output_dim))
        return output

class Userprelayer(nn.Module):
    def __init__(self, prefers, args):
        super(Userprelayer, self).__init__()
        self.embedding = nn.Embedding(prefers, args.userpre_embed_size)
        # self.dense = Dense(output_dim)
        self.output_dim = args.userpre_embed_size

    def forward(self, inputs):
        userpreembed = self.embedding(inputs)
        output = userpreembed.view((-1, self.output_dim))
        return output


class AttentionPooling(nn.Module):
    def __init__(self, d_h, hidden_size):
        super(AttentionPooling, self).__init__()
        self.att_fc1 = nn.Linear(d_h, hidden_size // 2)
        self.att_fc2 = nn.Linear(hidden_size // 2, 1)

    def forward(self, x, attn_mask=None):
        bz = x.shape[0]
        e = self.att_fc1(x)  # (bz, seq_len, 200)
        e = nn.Tanh()(e)
        alpha = self.att_fc2(e)  # (bz, seq_len, 1)

        alpha = torch.exp(alpha)
        if attn_mask is not None:
            alpha = alpha * attn_mask.unsqueeze(2)
        alpha = alpha / (torch.sum(alpha, dim=1, keepdim=True) + 1e-8)

        x = torch.bmm(x.permute(0, 2, 1), alpha)
        x = torch.reshape(x, (bz, -1))  # (bz, 400)
        return x


class NewsEncoder(nn.Module):
    def __init__(self, nums_word, preembed, categories, authors, prefers, args):
        super(NewsEncoder, self).__init__()
        self.titlelayer = TitleLayer(nums_word, preembed, args)
        self.categlayer = Categorylayer(categories, args)
        self.authorlayer = Authorlayer(authors, args)
        self.userprelayer = Userprelayer(prefers, args)
        self.attn = AttentionPooling(args.word_embed_size, args.word_embed_size // 2)
        self.bodylayer = None
        self.entitylayer = None
        self.title_size = args.title_size
        self.body_size = args.body_size
        self.batch_size = args.batch_size

    def forward(self, inputs, user_pre, expand_num):
        user_pre = torch.unsqueeze(user_pre, 1)
        user_pre = user_pre.expand(user_pre.size(0), expand_num)
        user_pre = user_pre.contiguous().view(inputs.size(0))

        userpre_embed = self.userprelayer(user_pre)
        title_embed = self.titlelayer(inputs[:, :self.title_size], userpre_embed)
        body_embed = self.titlelayer(inputs[:, self.title_size:self.title_size + self.body_size], userpre_embed)
        cata_embed = self.categlayer(inputs[:, -2])
        author_embed = self.authorlayer(inputs[:, -1])
        news_embed = torch.cat([title_embed, body_embed, cata_embed, author_embed], dim=-1).view(title_embed.size(0), -1,
                                                                                    title_embed.size(-1))
        # news_embed = torch.cat([title_embed, body_embed, cata_embed, author_embed], dim=-1)
        news_embed = self.attn(news_embed)

        # categ_embed = self.categlayer(inputs[:, self.title_size:])
        # news_embed = torch.cat((title_embed, categ_embed), 1)
        return news_embed


# class TimeDistributed(nn.Module):
#     def __init__(self, module, batch_first):
#         super(TimeDistributed, self).__init__()
#         self.module = module
#         self.batch_first = batch_first
#
#     def forward(self, input_seq):
#         assert len(input_seq.size()) > 2
#
#         # reshape input data --> (samples * timesteps, input_size)
#         # squash timesteps
#         reshaped_input = input_seq.contiguous().view(-1, input_seq.size(-1))
#
#         output = self.module(reshaped_input)
#         # We have to reshape Y
#         if self.batch_first:
#             # (samples, timesteps, output_size)
#             output = output.contiguous().view(input_seq.size(0), -1, output.size(-1))
#         else:
#             # (timesteps, samples, output_size)
#             output = output.contiguous().view(-1, input_seq.size(1), output.size(-1))
#         return output


class UserEncoder(nn.Module):
    def __init__(self, newscncoder, args):
        super(UserEncoder, self).__init__()
        self.dropout1 = nn.Dropout(args.droprate)
        self.dropout2 = nn.Dropout(args.droprate)

        self.newssize = args.num_heads * args.head_size

        self.multiatt = MultiHeadAttention(2, 25, 50)
        self.dense1 = nn.Linear(50, args.medialayer)
        self.dense2 = nn.Linear(args.medialayer, 1)
        self.newscncoder = newscncoder
        self.his_size = args.his_size
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def forward(self, inputs, user_pre):
        user_click, seq_len = inputs
        en = user_click.size(1)
        reshape_user_click = user_click.view(-1, user_click.size(-1))
        reshape_click_embed = self.newscncoder(reshape_user_click, user_pre, en)
        click_embed = reshape_click_embed.view(user_click.size(0), -1, reshape_click_embed.size(-1))
        click_embed = self.dropout1(click_embed)

        mask = torch.arange(0, self.his_size).to(self.device).unsqueeze(0).expand(user_click.size(0), self.his_size).lt(
            seq_len.unsqueeze(1)).float()
        mask = mask.masked_fill(mask == 0, -1e9)
        mask1 = torch.unsqueeze(torch.unsqueeze(mask, 1), 1)

        selfattn_output = self.multiatt(click_embed, mask1)
        selfattn_output = self.dropout2(selfattn_output)

        attention = self.dense1(selfattn_output)
        attention = self.dense2(attention)
        # mask2 = torch.unsqueeze(mask, 2)
        # attention += mask2
        attention_weight = F.softmax(attention, 1)
        output = torch.sum(attention_weight * selfattn_output, 1)

        return output


class Nrms(nn.Module):
    def __init__(self, nums_word, preembed, categories, authors, pre_len, args):
        super(Nrms, self).__init__()
        self.newsencoder = NewsEncoder(nums_word, preembed, categories, authors, pre_len, args)
        self.userencoder = UserEncoder(self.newsencoder, args)
        self.criterion = nn.CrossEntropyLoss()

    def forward(self, candidate_news, clicked_news, click_len, user_id, labels=None):
        en = candidate_news.size(1)
        reshape_candidate_news = candidate_news.view(-1, candidate_news.size(-1))
        reshape_news_embed = self.newsencoder(reshape_candidate_news, user_id, en)
        news_embed = reshape_news_embed.view(candidate_news.size(0), -1, reshape_news_embed.size(-1))
        user_embed = self.userencoder([clicked_news, click_len], user_id)
        # print(user_embed.shape)
        # print(user_embed)
        user_embed = torch.unsqueeze(user_embed, 2)
        # print(user_embed.shape)
        # print(user_embed)
        score = torch.squeeze(torch.matmul(news_embed, user_embed))
        # print(score.shape)
        # print(score)
        if labels is not None:
            # print(labels)
            loss = self.criterion(score, labels)
            return loss
        else:
            # score = F.softmax(score,0)
            score = F.sigmoid(score)
            return score


class Model(nn.Module):
    def __init__(self, preembed, args, logger, data):
        super(Model, self).__init__()
        args.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = Nrms(len(data.word_dict), preembed, len(data.categ_dict), len(data.author_dict), len(data.users), args).to(args.device)
        self.args = args
        self.logger = logger
        self.data = data

    def mtrain(self):
        args = self.args
        # train_sampler = RandomSampler(train_dataset)
        # train_dataloader = DataLoader(train_dataset,
        #                               sampler=train_sampler,
        #                               batch_size=args.batch_size)
        #
        batch_num = math.ceil(len(self.data.train_label) // args.batch_size)
        args.max_steps = args.epochs * batch_num
        # args.save_steps = len(train_dataloader) // 10
        # args.warmup_steps = len(train_dataloader)
        # args.logging_steps = len(self.data.train_label)

        if (args.optimizer == 'Adamw'):
            optimizer = torch.optim.AdamW(self.model.parameters(), lr=args.lr, eps=1e-8, weight_decay=args.l2)
        elif (args.optimizer == 'Adam'):
            optimizer = torch.optim.Adam(self.model.parameters(), lr=args.lr, weight_decay=args.l2)
        elif (args.optimizer == 'SGD'):
            optimizer = torch.optim.SGD(self.model.parameters(), lr=args.lr, weight_decay=args.l2)

        # scheduler = get_linear_schedule_with_warmup(optimizer,
        #                                             num_warmup_steps=int(args.max_steps * 0.2),
        #                                             num_training_steps=args.max_steps)

        args.n_gpu = torch.cuda.device_count()
        if args.n_gpu > 1:
            self.model = torch.nn.DataParallel(self.model)

        global_step = 0
        self.model.train()
        for epoch in range(args.epochs):
            train_loss = 0
            best_auc = 0
            start_train = time.time()
            train_progress = tqdm(enumerate(self.data.generate_batch_train_data()), dynamic_ncols=True,
                                  total=batch_num)
            for step, batch in train_progress:
                news, user_click, click_len, user_id, labels = (torch.LongTensor(x).to(args.device) for x in batch)
                del batch
                optimizer.zero_grad()
                loss = self.model(news, user_click, click_len, user_id, labels)
                if args.n_gpu > 1:
                    loss = loss.mean()
                loss.backward()
                if args.max_grad_norm != 0:
                    torch.nn.utils.clip_grad_norm(self.model.parameters(), args.max_grad_norm)
                train_loss += loss.item()
                optimizer.step()
                # scheduler.step()
                global_step += 1
                train_progress.set_description(u"[{}] Loss: {:,.6f} ----- ".format(epoch, train_loss / (step + 1)))

            self.logger.info('Time taken for training 1 epoch {} sec'.format(time.time() - start_train))
            self.logger.info('epoch:{}, loss:{}'.format(epoch, train_loss / batch_num))

            start_eval = time.time()
            preds = self.infer()
            auc, mrr, ndcg5, ndcg10 = self.getscore(preds, self.data.eval_label)
            self.logger.info('Time taken for testing 1 epoch {} sec'.format(time.time() - start_eval))
            self.logger.info('auc:{}, mrr:{}, ndcg5:{}, ndcg10:{}'.format(auc, mrr, ndcg5, ndcg10))

            if auc > best_auc:
                # test and save
                if args.save == 1:
                    model_to_save = self.model.module if hasattr(self.model,
                                                                 'module') else self.model  # Only save the model it-self
                    output_model_file = os.path.join(savepath, "pytorch_model.bin")
                    torch.save(model_to_save.state_dict(), output_model_file)

    def infer(self):
        args = self.args
        args.eval_batch_size = 1
        # eval_sampler = SequentialSampler(eval_dataset)
        # eval_dataloader = DataLoader(eval_dataset,
        #                              sampler=eval_sampler,
        #                              batch_size=1)
        # eval_loss = 0
        # self.model.eval()
        predict = []
        eval_progress = tqdm(enumerate(self.data.generate_batch_eval_data()), dynamic_ncols=True,
                             total=(len(self.data.eval_label) // args.eval_batch_size))
        for step, batch in eval_progress:
            news, user_click, click_len, user_id = (torch.LongTensor(x).to(args.device) for x in batch)
            with torch.no_grad():
                click_probability = self.model(news, user_click, click_len, user_id)
            predict.append(click_probability.cpu().numpy())

        return predict

    def getscore(self, preds, labels):
        aucs, mrrs, ndcg5s, ndcg10s = 0, 0, 0, 0
        testnum = len(labels)
        for i in range(testnum):
            aucs += roc_auc_score(labels[i], preds[i])
            mrrs += mrr_score(labels[i], preds[i])
            ndcg5s += ndcg_score(labels[i], preds[i], 5)
            ndcg10s += ndcg_score(labels[i], preds[i], 10)
        return aucs / testnum, mrrs / testnum, ndcg5s / testnum, ndcg10s / testnum


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('-g', '--gpu', help='set gpu device number 0-3', type=str, default='0')
    parser.add_argument('--note', help='model-note', type=str, default='None')
    parser.add_argument('--foldname', type=str, default='nrms')
    parser.add_argument('--test', help='test per epoch', type=int, default=1)
    parser.add_argument('-wd', '--word_embed_size', help='Embedding Size', type=int, default=50)
    parser.add_argument('-cd', '--categ_embed_size', help='Embedding Size', type=int, default=50)
    parser.add_argument('-ad', '--author_embed_size', help='Embedding Size', type=int, default=50)
    parser.add_argument('-up', '--userpre_embed_size', help='Embedding Size', type=int, default=25)
    parser.add_argument('--epochs', help='Max epoch', type=int, default=10)
    parser.add_argument('-n', '--neg_number', help='Negative Samples Count', type=int, default=1)
    parser.add_argument('-lr', help='learning_rate', type=float, default=0.002)  # or 1e-4
    parser.add_argument('-l2', help='l2 Regularization', type=float, default=0.0002)
    parser.add_argument('--dataset', help='path to file', type=str, default='small')
    parser.add_argument('-b', '--batch_size', help='Batch Size', type=int, default=128)
    parser.add_argument('--droprate', type=float, default=0.2)
    parser.add_argument('--num_heads', type=int, default=2)
    parser.add_argument('--head_size', type=int, default=25)
    parser.add_argument('--title_size', type=int, default=40)
    parser.add_argument('--body_size', type=int, default=60)
    parser.add_argument('--cata_size', type=int, default=1)
    parser.add_argument('--author_size', type=int, default=1)
    parser.add_argument('--his_size', type=int, default=50)
    parser.add_argument('--w2v', type=int, default=0)
    parser.add_argument('--medialayer', type=int, default=25)
    parser.add_argument('--max_grad_norm', type=int, default=0)
    parser.add_argument('--optimizer', type=str, default='Adam')
    parser.add_argument('--save', type=int, default=1)

    return parser.parse_args()


class MyDataset():
    def __init__(self, args):
        self.title_size = args.title_size
        self.body_size = args.body_size
        self.his_size = args.his_size
        self.batch_size = args.batch_size
        self.cata_size = args.cata_size
        self.author_size = args.author_size
        # self.user_info = np.load('user_sel.npz', allow_pickle=True)
        # self.user_sel = self.user_info['user_info'].tolist()
        # self.sel_top = self.user_info['sel_top'].tolist()
        if args.dataset == 'small':
            train_path = 'data/MINDsmall_train'
            test_path = 'data/MINDsmall_dev'
        elif args.dataset == 'large':
            train_path = '/media/omnisky/0857a6da-d9fb-4fc7-8572-de5e94175df3/xst/news_rec/MINDlarge_train'
            test_path = '/media/omnisky/0857a6da-d9fb-4fc7-8572-de5e94175df3/xst/news_rec/MINDlarge_dev'

        news_file = np.load('news_info.npz', allow_pickle=True)
        self.news = news_file['news'].tolist()
        self.newsidenx = news_file['newsidenx'].tolist()
        self.word_dict = news_file['word_dict'].tolist()
        self.categ_dict = news_file['categ_dict'].tolist()
        self.author_dict = news_file['author_dict'].tolist()
        self.news_features = news_file['news_features'].tolist()

        train_test_data = np.load('train_test_data_4.2_n10.npz', allow_pickle=True)
        self.negnums = train_test_data['negnums'].tolist()
        self.train_user_his = train_test_data['train_user_his'].tolist()
        self.train_candidate = train_test_data['train_candidate'].tolist()
        self.train_label = train_test_data['train_label'].tolist()
        self.train_his_len = train_test_data['train_his_len'].tolist()
        self.ori_his_size = train_test_data['ori_his_size']
        self.users = train_test_data['users'].tolist()
        self.train_user_id = train_test_data['train_user_id'].tolist()

        self.eval_candidate = train_test_data['eval_candidate'].tolist()
        self.eval_label = train_test_data['eval_label'].tolist()
        self.eval_user_his = train_test_data['eval_user_his'].tolist()
        self.eval_click_len = train_test_data['eval_click_len'].tolist()
        self.eval_user_id = train_test_data['eval_user_id'].tolist()

        # news_path = '/news_train'
        # user_path = '/behaviors_train1'
        # with open(train_path + news_path, 'r', encoding='utf-8') as f:
        #     trainnewsfile = f.readlines()
        # with open(train_path + user_path, 'r', encoding='utf-8') as f:
        #     trainuserfile = f.readlines()
        # # with open(test_path + news_path, 'r', encoding='utf-8') as f:
        # #     testnewsfile = f.readlines()
        # user_path = '/behaviors_test1'
        # with open(test_path + user_path, 'r', encoding='utf-8') as f:
        #     testuserfile = f.readlines()
        #
        # self.news = {}
        # jieba.load_userdict('dict.txt')
        # for line in trainnewsfile:
        #     # print(line)
        #     linesplit = line.split('\t')
        #     # print(linesplit[0])
        #     # if len(linesplit[0]) > 10:
        #     #     continue
        #     if len(linesplit) < 5:
        #         if linesplit[0] != '':
        #             self.news[linesplit[0]] = ['', '', '', '']
        #         continue
        #     # self.news[linesplit[0]] = [linesplit[1].strip(), linesplit[2].strip()]
        #     self.news[linesplit[0]] = []
        #     fenci = jieba.cut(linesplit[3][:40])
        #     result = []
        #     for wd in fenci:
        #         result.append(wd)
        #     self.news[linesplit[0]].append(result)
        #     fenci = jieba.cut(linesplit[4][:60])
        #     result = []
        #     for wd in fenci:
        #         result.append(wd)
        #     self.news[linesplit[0]].append(result)
        #     self.news[linesplit[0]].append(linesplit[1].strip())
        #     self.news[linesplit[0]].append(linesplit[2].strip())
        #     # print((linesplit[1].strip(),linesplit[2].strip(),word_tokenize(linesplit[3].lower())))
        #
        # # for line in testnewsfile:
        # #     # print(line)
        # #     linesplit = line.split('\t')
        # #     self.news[linesplit[0]] = (linesplit[1].strip(), linesplit[2].strip(), word_tokenize(linesplit[3].lower()))
        #
        # # self.ori_title_size = max([len(x) for x in self.news.values()])
        #
        # self.newsidenx = {'NULL': 0}
        # nid = 1
        # for id in self.news:
        #     self.newsidenx[id] = nid
        #     nid += 1
        # # print(newsidenx)
        #
        # self.word_dict = {'PADDING': 0}
        # self.categ_dict = {'PADDING': 0}
        # self.author_dict = {'PADDING': 0}
        # self.news_features = [[0] * (self.title_size + self.body_size + self.cata_size + self.author_size)]
        # self.sentence = []
        # for newid in self.news:
        #     title = []
        #     body = []
        #     features = self.news[newid]
        #     if features[2] not in self.categ_dict:
        #         self.categ_dict[features[2]] = len(self.categ_dict)
        #     # if features[1] not in self.categ_dict:
        #     #     self.categ_dict[features[1]] = len(self.categ_dict)
        #     if features[3] not in self.author_dict:
        #         self.author_dict[features[3]] = len(self.author_dict)
        #     for w in features[0]:
        #         if w not in self.word_dict:
        #             self.word_dict[w] = len(self.word_dict)
        #         title.append(self.word_dict[w])
        #
        #     title = title[:self.title_size]
        #     title = title + [0] * (self.title_size - len(title))
        #
        #     for w in features[1]:
        #         if w not in self.word_dict:
        #             self.word_dict[w] = len(self.word_dict)
        #         body.append(self.word_dict[w])
        #     body = body[:args.body_size] + [0] * (args.body_size - len(body))
        #     title.extend(body)
        #     title.append(self.categ_dict[features[2]])
        #     title.append(self.author_dict[features[3]])
        #
        #     # title.append(self.categ_dict[features[0]])
        #     # title.append(self.categ_dict[features[1]])
        #     self.news_features.append(title)
        #
        # self.negnums = args.neg_number
        # self.train_user_his = []
        # self.train_candidate = []
        # self.train_label = []
        # self.train_his_len = []
        # self.train_user_sel_can = []
        # self.train_user_sel_his = []
        # self.ori_his_size = 0
        # for line in trainuserfile:
        #     # print(line)
        #     linesplit = line.split('\t')
        #     user_id = int(linesplit[1])
        #     clickids = [n for n in linesplit[3].split(' ') if n != '']
        #     self.ori_his_size = max(self.ori_his_size, len(clickids))
        #     clickids = clickids[-self.his_size:]
        #     click_len = len(clickids)
        #     # print(click_len)
        #     clickids = clickids + ['NULL'] * (self.his_size - len(clickids))
        #     clickids = [self.newsidenx[n] for n in clickids]
        #     # print(clickids)
        #
        #     pnew = []
        #     nnew = []
        #     for candidate in linesplit[4].split(' '):
        #         candidate = candidate.strip().split('-')
        #         if (candidate[1] == '1'):
        #             pnew.append(self.newsidenx[candidate[0]])
        #         else:
        #             nnew.append(self.newsidenx[candidate[0]])
        #     # print(pnew,nnew)
        #     if len(nnew) == 0:
        #         continue
        #     for pos in pnew:
        #
        #         if (self.negnums > len(nnew)):
        #             negsam = random.sample(nnew * ((self.negnums // len(nnew)) + 1), self.negnums)
        #         else:
        #             negsam = random.sample(nnew, self.negnums)
        #
        #         negsam.append(pos)
        #         # shuffle
        #         self.train_candidate.append(negsam)
        #         self.train_label.append(self.negnums)
        #         self.train_user_his.append(clickids)
        #         can_user_pre = []
        #         for j in range(len(negsam)):
        #             can_user_pre.append(self.user_sel[user_id])
        #         self.train_user_sel_can.append(can_user_pre)
        #         his_user_pre = []
        #         for j in range(len(clickids)):
        #             his_user_pre.append(self.user_sel[user_id])
        #         self.train_user_sel_his.append(his_user_pre)
        #         self.train_his_len.append(click_len)
        #
        # self.eval_candidate = []
        # self.eval_label = []
        # self.eval_user_his = []
        # self.eval_user_sel_can = []
        # self.eval_user_sel_his = []
        # self.eval_click_len = []
        #
        # for line in testuserfile:
        #     linesplit = line.split('\t')
        #     user_id = int(linesplit[1])
        #     clickids = [n for n in linesplit[3].split(' ') if n != '']
        #     clickids = clickids[-self.his_size:]
        #     click_len = len(clickids)
        #     clickids = clickids + ['NULL'] * (self.his_size - len(clickids))
        #     clickids = [self.newsidenx[n] for n in clickids]
        #     # print(clickids)
        #     temp = []
        #     temp_label = []
        #     if len(linesplit[4].split(' ')) < 2:
        #         continue
        #     for candidate in linesplit[4].split(' '):
        #         candidate = candidate.strip().split('-')
        #         temp.append(self.newsidenx[candidate[0]])
        #         temp_label.append(int(candidate[1]))
        #     # temp=temp + [0]*(295-len(temp))
        #     # temp_label = temp_label + [-1]*(295-len(train_label))
        #
        #     # pos = temp[-1]
        #     # temp = temp[:-1]
        #     # temp = random.sample(temp, min(10, len(temp)))
        #     # temp_label = [0] * len(temp)
        #     # temp.append(pos)
        #     # temp_label.append(1)
        #     self.eval_candidate.append(temp)
        #     self.eval_label.append(temp_label)
        #     self.eval_user_his.append(clickids)
        #     self.eval_click_len.append(click_len)
        #     can_user_pre = []
        #     for j in range(len(temp)):
        #         can_user_pre.append(self.user_sel[user_id])
        #     self.eval_user_sel_can.append(can_user_pre)
        #     his_user_pre = []
        #     for j in range(len(clickids)):
        #         his_user_pre.append(self.user_sel[user_id])
        #     self.eval_user_sel_his.append(his_user_pre)

        self.train_candidate = np.array(self.train_candidate, dtype='int32')
        self.train_label = np.array(self.train_label, dtype='int32')
        self.train_user_his = np.array(self.train_user_his, dtype='int32')
        self.train_his_len = np.array(self.train_his_len, dtype='int32')
        self.train_user_id = np.array(self.train_user_id, dtype='int32')
        self.news_features = np.array(self.news_features)
        # self.train_candidate = torch.tensor(self.train_candidate, dtype=torch.int32)
        # self.train_label = torch.tensor(self.train_label, dtype=torch.int32)
        # self.train_user_his = torch.tensor(self.train_user_his, dtype=torch.int32)
        # self.train_his_len = torch.tensor(self.train_his_len, dtype=torch.int32)
        # self.news_features = torch.tensor(self.news_features)
        # print(self.news_title)

        # max_impression = max([len(x) for x in self.eval_candidate])
        # print('max_impression:', max_impression)

    def generate_batch_train_data(self):
        idlist = np.arange(len(self.train_label))
        np.random.shuffle(idlist)
        batches = [idlist[range(self.batch_size * i, min(len(self.train_label), self.batch_size * (i + 1)))] for i in
                   range(len(self.train_label) // self.batch_size + 1)]
        for i in batches:
            # print(i)
            # print(self.train_candidate[i])
            # print(self.train_user_his[i])
            # print(self.news_features[self.train_candidate[i]])
            # print(self.news_features[self.train_user_his[i]])
            # if len(i) != self.batch_size:
            #     continue
            item = self.news_features[self.train_candidate[i]]
            # print(item)
            user = self.news_features[self.train_user_his[i]]
            # print(user)
            user_len = self.train_his_len[i]
            train_user_id = self.train_user_id[i]
            # print('------------------',user_len)
            # print(i)

            yield (item, user, user_len, train_user_id, self.train_label[i])

    def generate_batch_eval_data(self):
        for i in range(len(self.eval_candidate)):
            news = [self.news_features[self.eval_candidate[i]]]
            user = [self.news_features[self.eval_user_his[i]]]
            user_len = [self.eval_click_len[i]]
            test_label = self.eval_label[i]
            eval_user_id = [self.eval_user_id[i]]

            yield (news, user, user_len, eval_user_id)


if __name__ == '__main__':
    args = parse_args()
    savepath = 'result/' + args.foldname + '/' + time.strftime('%m_%d-%H-%M-%S', time.localtime(time.time()))
    log = Logger('root', savepath)
    logger = log.getlog()
    write_para = ''
    for k, v in vars(args).items():
        write_para += '\n' + k + ' : ' + str(v)
    logger.info('\n' + write_para + '\n')

    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    data = MyDataset(args)
    model = Model(None, args, logger, data)
    model.mtrain()

