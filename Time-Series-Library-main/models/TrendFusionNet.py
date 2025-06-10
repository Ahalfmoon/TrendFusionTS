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

        # get parameters
        self.seq_len = configs.seq_len
        self.enc_in = configs.enc_in
        self.d_model = configs.d_model
        self.dropout = configs.dropout

        self.task_name = configs.task_name
        if self.task_name == 'classification' or self.task_name == 'anomaly_detection' or self.task_name == 'imputation':
            self.pred_len = configs.seq_len
        else:
            self.pred_len = configs.pred_len

        self.seg_len = configs.seg_len
        self.seg_num_x = self.seq_len // self.seg_len
        self.seg_num_y = self.pred_len // self.seg_len

        # building model
        self.valueEmbedding = nn.Sequential(
            nn.Linear(self.seg_len, self.d_model),
            nn.ReLU()
        )
        # 特征映射层，将输入特征映射到隐藏空间
        input_dim = 3
        hidden_dim =  64 #32#64
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
        # b:batch_size c:channel_size s:seq_len s:seq_len
        # d:d_model w:seg_len n:seg_num_x m:seg_num_y
        batch_size = x.size(0)

        # normalization and permute     b,s,c -> b,c,s
        ##seq_last = x[:, -1:, :].detach()
        seq_last = x[:, -1:, 0].detach()  # new try
        seq_last = seq_last.unsqueeze(2)  # new try

        x = (x - seq_last).permute(0, 2, 1) # b,c,s
        #print("xxxxx?", x.shape, x) #64, 3, 720]
        x_core = x[:, 0, :]  # b,96
        x_fea = x[:, 1:, :]
        #print(x_core.shape,x_fea.shape) #torch.Size([32, 720]) torch.Size([32, 6, 720])
        x_core = x_core.reshape(batch_size, self.seg_num_x, self.seg_len)  # new try（16,16,6）
        x_fea = x_fea.reshape(batch_size, self.seg_num_x, self.seg_len * 2)  # （16,16,6*2）
        x_fea_1 = x_fea[:, :, :self.seg_len]  # 6 # 第一个维度
        x_fea_2 = x_fea[:, :, self.seg_len:]  # 第二个维度

        #print("x_core?",x_core.shape,x) #[64, 15, 48]

        encoded_x_core = self.valueEmbedding(x_core)  # x.view(-1, self.seg_num_x, self.seg_len) # origin   new new
        #print("xxx ?", encoded_x_core.shape) #b,16,64  #64, 15, 512]

        # 分别对这两个部分应用 self.valueEmbedding
        encoded_part1 = self.Embedding(x_fea_1)  # 形状为 (16, 16, 32)
        # print("xxx 1 ?", encoded_part1.shape)
        encoded_part2 = self.Embedding(x_fea_2)  # 形状为 (16, 16, 32)
        # print("xxx 2?", encoded_part1.shape)
        # 在隐藏层维度上拼接起来，得到最终的形状 (16, 16, 64)
        fea_output = torch.cat((encoded_part1, encoded_part2), dim=2)

        x = torch.cat((encoded_x_core, fea_output), dim=2)
        #x = encoded_x_core
        # segment and embedding    b,c,s -> bc,n,w -> bc,n,d
        #x = self.valueEmbedding(x.reshape(-1, self.seg_num_x, self.seg_len))

        # encoding
        _, hn = self.rnn0(x) # bc,n,d  1,bc,d

        # m,d//2 -> 1,m,d//2 -> c,m,d//2
        # c,d//2 -> c,1,d//2 -> c,m,d//2
        # c,m,d -> cm,1,d -> bcm, 1, d
        pos_emb = torch.cat([
            self.pos_emb.unsqueeze(0).repeat(self.enc_in, 1, 1),
            self.channel_emb.unsqueeze(1).repeat(1, self.seg_num_y, 1)
        ], dim=-1).view(-1, 1, self.d_model).repeat(batch_size,1,1)

        _, hy = self.rnn(pos_emb, hn.repeat(1, 1, self.seg_num_y).view(1, -1, self.d_model)) # bcm,1,d  1,bcm,d

        # 1,bcm,d -> 1,bcm,w -> b,c,s
        y = self.predict(hy).view(-1, self.enc_in, self.pred_len)

        # permute and denorm
        y = y.permute(0, 2, 1) + seq_last
        return y

    def forecast(self, x_enc):
        # Encoder
        return self.encoder(x_enc)

    def imputation(self, x_enc):
        # Encoder
        return self.encoder(x_enc)

    def anomaly_detection(self, x_enc):
        # Encoder
        return self.encoder(x_enc)

    def classification(self, x_enc):
        # Encoder
        enc_out = self.encoder(x_enc)
        # Output
        # (batch_size, seq_length * d_model)
        output = enc_out.reshape(enc_out.shape[0], -1)
        # (batch_size, num_classes)
        output = self.projection(output)
        return output

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None):
        if self.task_name == 'long_term_forecast' or self.task_name == 'short_term_forecast':
            dec_out = self.forecast(x_enc)
            return dec_out[:, -self.pred_len:, :]  # [B, L, D]
        if self.task_name == 'imputation':
            dec_out = self.imputation(x_enc)
            return dec_out  # [B, L, D]
        if self.task_name == 'anomaly_detection':
            dec_out = self.anomaly_detection(x_enc)
            return dec_out  # [B, L, D]
        if self.task_name == 'classification':
            dec_out = self.classification(x_enc)
            return dec_out  # [B, N]
        return None
