import torch
import torch.nn as nn
import zipfile
import numpy as np

# This function is now defined outside of the DanModel class
def load_embedding(vocab, emb_file, emb_size):
    emb_matrix = np.zeros((len(vocab), emb_size))
    with open(emb_file, 'r', encoding='utf-8') as f:
        for line in f:
            values = line.split()
            word = values[0]
            vector = np.asarray(values[1:], "float32")
            if word in vocab.word2id:
                idx = vocab.word2id[word]
                emb_matrix[idx] = vector
    return emb_matrix




class BaseModel(nn.Module):
    def __init__(self, args, vocab, tag_size):
        super(BaseModel, self).__init__()
        self.args = args
        self.vocab = vocab
        self.tag_size = tag_size

    def save(self, path):
        # Save model
        print(f'Saving model to {path}')
        ckpt = {
            'args': self.args,
            'vocab': self.vocab,
            'state_dict': self.state_dict()
        }
        torch.save(ckpt, path)

    def load(self, path):
        # Load model
        print(f'Loading model from {path}')
        ckpt = torch.load(path)
        self.vocab = ckpt['vocab']
        self.args = ckpt['args']
        self.load_state_dict(ckpt['state_dict'])


# def load_embedding(self, vocab, emb_file, emb_size):
#     emb_matrix = np.zeros((len(vocab), emb_size))
#     with open(emb_file, 'r', encoding='utf-8') as f:
#         for line in f:
#             values = line.split()
#             word = values[0]
#             vector = np.asarray(values[1:], "float32")
#             if word in vocab.word2id:
#                 idx = vocab.word2id[word]
#                 emb_matrix[idx] = vector
#     return emb_matrix


class DanModel(BaseModel):
    def __init__(self, args, vocab, tag_size):
        super(DanModel, self).__init__(args, vocab, tag_size)
        self.define_model_parameters()
        self.init_model_parameters()

        # Use pre-trained word embeddings if emb_file exists
        if args.emb_file is not None:
            # Directly call the standalone load_embedding function
            embedding_matrix = load_embedding(self.vocab, args.emb_file, args.emb_size)
            self.copy_embedding_from_numpy(embedding_matrix)


    def define_model_parameters(self):
        vocab_size = len(self.vocab)  # Assuming self.vocab provides the total vocabulary size
        self.embedding = nn.Embedding(vocab_size, self.args.emb_size)
        self.feedforward_layers = nn.Sequential(
            nn.Linear(self.args.emb_size, self.args.hid_size),
            nn.ReLU(),
            nn.Dropout(self.args.emb_drop),
            *[nn.Sequential(nn.Linear(self.args.hid_size, self.args.hid_size), nn.ReLU(), nn.Dropout(self.args.hid_drop)) for _ in range(self.args.hid_layer - 1)],
        )
        self.output_layer = nn.Linear(self.args.hid_size, self.tag_size)


    def init_model_parameters(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.constant_(m.bias, 0)


    def copy_embedding_from_numpy(self, embedding_matrix):
        self.embedding.weight.data.copy_(torch.from_numpy(embedding_matrix))
        self.embedding.weight.requires_grad = False  # Optionally freeze embeddings



    def forward(self, x):
        # x: [batch_size, seq_length]
        embeddings = self.embedding(x)  # [batch_size, seq_length, emb_size]
        # Average embeddings along the sequence length dimension
        averaged_embeddings = embeddings.mean(dim=1)  # [batch_size, emb_size]
        hidden = self.feedforward_layers(averaged_embeddings)  # [batch_size, hid_size]
        scores = self.output_layer(hidden)  # [batch_size, tag_size]
        return scores

