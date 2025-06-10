import torch
import torch.nn as nn
import torch.nn.functional as F
from layers.Autoformer_EncDec import series_decomp


class Model(nn.Module):
    """
    Paper link: https://arxiv.org/abs/2308.11200.pdf
    """

    def __init__(self, configs):
        super(Model, self).__init__()

        # 获取参数
        self.seq_len = configs.seq_len
        self.enc_in = configs.enc_in
        self.d_model = configs.d_model
        self.dropout = configs.dropout

        self.task_name = configs.task_name
        if self.task_name in ['classification', 'anomaly_detection', 'imputation']:
            self.pred_len = configs.seq_len
        else:
            self.pred_len = configs.pred_len

        self.seg_len = configs.seg_len
        self.seg_num_x = self.seq_len // self.seg_len
        self.seg_num_y = self.pred_len // self.seg_len

        # 构建模型
        self.valueEmbedding = nn.Sequential(
            nn.Linear(self.seg_len, self.d_model),
            nn.ReLU()
        )
        # 特征映射层，将输入特征映射到隐藏空间
        hidden_dim = 64
        self.Embedding = nn.Sequential(
            nn.Linear(self.seg_len, hidden_dim),
            nn.ReLU()
        )
        self.rnn0 = nn.GRU(input_size=self.d_model, hidden_size=self.d_model, num_layers=1, bias=True,
                          batch_first=True, bidirectional=False)
        self.rnn = nn.GRU(input_size=self.d_model, hidden_size=self.d_model, num_layers=1, bias=True,
                              batch_first=True, bidirectional=False)
        self.pos_emb = nn.Parameter(torch.randn(self.seg_num_y, self.d_model // 2))
        self.channel_emb = nn.Parameter(torch.randn(self.enc_in, self.d_model // 2))

        self.predict = nn.Sequential(
            nn.Dropout(self.dropout),
            nn.Linear(self.d_model, self.seg_len)
        )

        if self.task_name == 'classification':
            self.act = F.gelu
            self.dropout = nn.Dropout(configs.dropout)
            self.projection = nn.Linear(
                configs.enc_in * configs.seq_len, configs.num_class)

    def encoder(self, x):
        # b:batch_size c:channel_size s:seq_len
        # d:d_model w:seg_len n:seg_num_x m:seg_num_y
        batch_size = x.size(0)

        # 归一化和转置     b,s,c -> b,c,s
        seq_last = x[:, -1:, 0].detach().unsqueeze(2)

        x = (x - seq_last).permute(0, 2, 1) # b,c,s
        x_core = x[:, 0, :]
        x_fea = x[:, 1:, :]
        x_core = x_core.reshape(batch_size, self.seg_num_x, self.seg_len)
        x_fea = x_fea.reshape(batch_size, self.seg_num_x, self.seg_len * 2)
        x_fea_1 = x_fea[:, :, :self.seg_len]
        x_fea_2 = x_fea[:, :, self.seg_len:]

        encoded_x_core = self.valueEmbedding(x_core)

        # 分别对这两个部分应用 self.valueEmbedding
        encoded_part1 = self.Embedding(x_fea_1)
        encoded_part2 = self.Embedding(x_fea_2)
        fea_output = torch.cat((encoded_part1, encoded_part2), dim=2)

        x = torch.cat((encoded_x_core, fea_output), dim=2)

        # 编码
        _, hn = self.rnn0(x)

        pos_emb = torch.cat([
            self.pos_emb.unsqueeze(0).repeat(self.enc_in, 1, 1),
            self.channel_emb.unsqueeze(1).repeat(1, self.seg_num_y, 1)
        ], dim=-1).view(-1, 1, self.d_model).repeat(batch_size,1,1)

        _, hy = self.rnn(pos_emb, hn.repeat(1, 1, self.seg_num_y).view(1, -1, self.d_model))

        y = self.predict(hy).view(-1, self.enc_in, self.pred_len)

        # 转置和反归一化
        y = y.permute(0, 2, 1) + seq_last
        return y

    def forecast(self, x_enc):
        return self.encoder(x_enc)

    def imputation(self, x_enc):
        return self.encoder(x_enc)

    def anomaly_detection(self, x_enc):
        return self.encoder(x_enc)

    def classification(self, x_enc):
        enc_out = self.encoder(x_enc)
        output = enc_out.reshape(enc_out.shape[0], -1)
        output = self.projection(output)
        return output

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None):
        if self.task_name in ['long_term_forecast', 'short_term_forecast']:
            dec_out = self.forecast(x_enc)
            return dec_out[:, -self.pred_len:, :]
        if self.task_name == 'imputation':
            dec_out = self.imputation(x_enc)
            return dec_out
        if self.task_name == 'anomaly_detection':
            dec_out = self.anomaly_detection(x_enc)
            return dec_out
        if self.task_name == 'classification':
            dec_out = self.classification(x_enc)
            return dec_out
        return None
