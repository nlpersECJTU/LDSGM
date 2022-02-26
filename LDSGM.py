import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import RobertaModel
from Layers.gcn import GraphConvolution
import pickle


class Label2Context(nn.Module):  # label attention in the encoder
    def __init__(self, label_embedding_size, attn_hidden_size):
        super(Label2Context,self).__init__()
        self.proj_label_top = nn.Linear(label_embedding_size, attn_hidden_size, bias=False)
        self.proj_label_sec = nn.Linear(label_embedding_size, attn_hidden_size, bias=False)
        self.proj_label_conn = nn.Linear(label_embedding_size, attn_hidden_size, bias=False)

    def forward(self, input_data, label_repr, use_label_num):
        seq_len = input_data.shape[1]
        lb_num = use_label_num[1] - use_label_num[0]
        embedding_label = label_repr[use_label_num[0]:use_label_num[1]]
        if lb_num == 4:
            embedding_label = self.proj_label_top(embedding_label)
        if lb_num == 11:
            embedding_label = self.proj_label_sec(embedding_label)
        if lb_num == 102:
            embedding_label = self.proj_label_conn(embedding_label)
        embedding_label = embedding_label.transpose(0, 1)
        input_data = F.normalize(input_data, dim=-1, p=2)
        embedding_label = F.normalize(embedding_label, dim=0, p=2)
        G = torch.matmul(input_data.reshape(-1, input_data.shape[-1]), embedding_label)
        G = G.reshape(-1, seq_len, G.shape[-1])
        max_G = G.max(-1, keepdim=True)[0]  # [batch_size, seq_length, 1]
        soft_max_G = torch.softmax(max_G, dim=1)
        output = torch.sum(input_data * soft_max_G, axis=1)
        return output


class Decoder(nn.Module):
    def __init__(self, label_embedding_size, hidden_size, enc_hidden_size, dec_hidden_size):
        super().__init__()
        self.hidden_size = hidden_size
        self.rnn = nn.GRU(enc_hidden_size+100, dec_hidden_size)
        self.proj_out = nn.Linear(dec_hidden_size*2+100, dec_hidden_size)

    def forward(self, input, hidden, encoder_outputs):
        attention_outputs = encoder_outputs
        rnn_input = torch.cat([attention_outputs, input], dim=-1)
        output, hidden = self.rnn(rnn_input.unsqueeze(0), hidden.unsqueeze(0))

        assert (output == hidden).all()
        output = output.squeeze(0)
        hidden = hidden.squeeze(0)
        pred = self.proj_out(torch.cat([attention_outputs, output, input], dim=1))
        return pred, hidden


class Model(nn.Module):
    def __init__(self, config):
        super(Model, self).__init__()
        with open('label_graph.g', 'rb') as f:
            label_graph = pickle.load(f)
        self.config = config
        self.label_graph = torch.from_numpy(label_graph).float().to(config.device)
        self.label_embedding = nn.Parameter(torch.randn(config.label_num, config.label_embedding_size, dtype=torch.float32))
        nn.init.xavier_normal_(self.label_embedding.data)

        self.bert = RobertaModel.from_pretrained(config.bert_path)
        for param in self.bert.parameters():
            param.requires_grad = config.finetune_bert

        self.gcn_layers = nn.ModuleList()
        for i in range(config.num_gcn_layer):
            self.gcn_layers.append(GraphConvolution(config.label_embedding_size, config.label_embedding_size))

        # top-down manner Decoder
        self.decoder = Decoder(config.label_embedding_size, config.attn_hidden_size, config.enc_hidden_size, config.dec_hidden_size)
        # bottom-up manner auxilary Decoder
        self.decoder_reverse = Decoder(config.label_embedding_size, config.attn_hidden_size, config.enc_hidden_size,config.dec_hidden_size)
        self.label2context = Label2Context(config.label_embedding_size, config.attn_hidden_size)
        self.fc1_top = nn.Linear(config.dec_hidden_size, config.n_top)
        self.fc1_sec = nn.Linear(config.dec_hidden_size, config.n_sec)
        self.fc1_conn = nn.Linear(config.dec_hidden_size, config.n_conn)
        self.fc2_top = nn.Linear(config.dec_hidden_size, config.n_top)
        self.fc2_sec = nn.Linear(config.dec_hidden_size, config.n_sec)
        self.fc2_conn = nn.Linear(config.dec_hidden_size, config.n_conn)
        self.softmax = nn.Softmax(dim=1)
        self.dropout = nn.Dropout(0.2)

    def forward(self, x, argmask):
        context = x[0]  # 
        mask = x[2]
        bert_out = self.bert(context, attention_mask=mask)
        hidden_last, pooled, hidden_all = bert_out.last_hidden_state, bert_out.pooler_output, bert_out.hidden_states
        hidden_cls, encode_out = hidden_last[:, 0, :], hidden_last[:, 1:, :]
        encode_out = self.dropout(encode_out)
        for gcn_layer in self.gcn_layers:
            label_repr = F.relu(gcn_layer(self.label_embedding, self.label_graph))
        bs = hidden_last.shape[0]
        hidden_g = hidden_cls
        hidden = hidden_cls
        # input = torch.FloatTensor(torch.zeros(bs, self.label_embedding.shape[1])).to(self.config.device)
        # input_reverse = torch.FloatTensor(torch.zeros(bs, self.label_embedding.shape[1])).to(self.config.device)
        input = label_repr[0: 4].sum(dim=0, keepdim=True).repeat(bs, 1)
        input_reverse = label_repr[15: 117].sum(dim=0, keepdim=True).repeat(bs, 1)
        hidden_reverse = hidden_cls
        out_top, out_sec, out_conn = 0, 0, 0
        for i in range(3):
            if i == 0:  # top-level prediction
                label2context_out = self.label2context(encode_out, label_repr, [0, 4])
                pred, output = self.decoder(input, hidden.contiguous(), label2context_out)
                hidden = output
                out_top = self.fc1_top(pred)
                t = self.softmax(out_top).unsqueeze(-1).expand(out_top.shape[0], out_top.shape[1], label_repr.shape[1])
                input = (t * label_repr[0:4]).sum(dim=1, keepdim=False)

            if i == 1:  # second-level prediction
                hidden = hidden + hidden_g
                label2context_out = self.label2context(encode_out, label_repr, [4, 15])
                pred, output = self.decoder(input, hidden.contiguous(), label2context_out)
                hidden = output
                out_sec = self.fc1_sec(pred)
                t = self.softmax(out_sec).unsqueeze(-1).expand(out_sec.shape[0], out_sec.shape[1], label_repr.shape[1])
                input = (t * label_repr[4:15]).sum(dim=1, keepdim=False)

            if i == 2:  # conn-level prediction
                hidden = hidden + hidden_g
                label2context_out = self.label2context(encode_out, label_repr, [15, 117])
                pred, output = self.decoder(input, hidden.contiguous(), label2context_out)
                out_conn = self.fc1_conn(pred)
                t = self.softmax(out_conn).unsqueeze(-1).expand(out_conn.shape[0], out_conn.shape[1],
                                                               label_repr.shape[1])
                input = (t * label_repr[15:117]).sum(dim=1, keepdim=False)
        # the bottom-up manner auxilary decoder
        for i in range(3):
            if i == 0:
                hidden_reverse = hidden_reverse.contiguous()
                label2context_out = self.label2context(encode_out, label_repr, [15, 117])
                pred_reverse, output_reverse = self.decoder_reverse(input_reverse, hidden_reverse,label2context_out)
                hidden_reverse = output_reverse
                out_conn_reverse = self.fc2_conn(pred_reverse)
                t = self.softmax(out_conn_reverse).unsqueeze(-1).expand(out_conn_reverse.shape[0], out_conn_reverse.shape[1], label_repr.shape[1])
                input_reverse = (t * label_repr[15:117]).sum(dim=1, keepdim=False)

            if i == 1:
                hidden_reverse = hidden_reverse + hidden_g
                hidden_reverse = hidden_reverse.contiguous()
                label2context_out = self.label2context(encode_out, label_repr, [4, 15])
                pred_reverse, output_reverse = self.decoder_reverse(input_reverse, hidden_reverse, label2context_out)
                hidden_reverse = output_reverse
                out_sec_reverse = self.fc2_sec(pred_reverse)
                t = self.softmax(out_sec_reverse).unsqueeze(-1).expand(out_sec_reverse.shape[0],
                                                                       out_sec_reverse.shape[1], label_repr.shape[1])
                input_reverse = (t * label_repr[4:15]).sum(dim=1, keepdim=False)

            if i == 2:
                hidden_reverse = hidden_reverse + hidden_g
                hidden_reverse = hidden_reverse.contiguous()
                label2context_out = self.label2context(encode_out, label_repr, [0, 4])
                pred_reverse, output_reverse = self.decoder_reverse(input_reverse, hidden_reverse, label2context_out)
                out_top_reverse = self.fc2_top(pred_reverse)
        return out_top, out_sec, out_conn, out_top_reverse, out_sec_reverse, out_conn_reverse
