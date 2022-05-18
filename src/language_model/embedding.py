from bio_embeddings import embed
import numpy as np
import pandas as pd
import torch


class LMEmbedder:
    
    def __init__(self, device):
        self.embedder = None
        self.generator = None
        self.device = device

    def init_embedder(self):
        del self.embedder, self.generator, 
        self.embedder = None
        self.generator = None
    
    def albert_embedding(self, sequences):
        self.embedder = embed.ProtTransAlbertBFDEmbedder(device=self.device)
        self.generator = self.embedder.embed_many(sequences)
        features = np.array([np.mean(v, axis=0) for v in self.generator])
        self.init_embedder()
        return pd.DataFrame(features)
    
    def bert_embedding(self, sequences):
        self.embedder = embed.ProtTransBertBFDEmbedder(device=self.device)
        self.generator = self.embedder.embed_many(sequences)
        features = np.array([np.mean(v, axis=0) for v in self.generator])
        self.init_embedder()
        return pd.DataFrame(features)
    
    def t5_embedding(self, sequences):
        self.embedder = embed.ProtTransT5BFDEmbedder(device=self.device)
        self.generator = self.embedder.embed_many(sequences)
        features = np.array([np.mean(v, axis=0) for v in self.generator])
        self.init_embedder()
        return pd.DataFrame(features)
    
    def t5ft_embedding(self, sequences):
        self.embedder = embed.ProtTransT5XLU50Embedder(device=self.device)
        self.generator = self.embedder.embed_many(sequences)
        features = np.array([np.mean(v, axis=0) for v in self.generator])
        self.init_embedder()
        return pd.DataFrame(features)
    
    def xlnet_embedding(self, sequences):
        self.embedder = embed.ProtTransXLNetUniRef100Embedder()
        print(self.embedder)
        self.generator = self.embedder.embed_many(sequences)
        features = np.array([np.mean(v, axis=0) for v in self.generator])
        self.init_embedder()
        return pd.DataFrame(features)