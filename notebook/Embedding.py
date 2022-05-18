from bio_embeddings import embed
import numpy as np


class Embedder:
    
    def __init__(self):
        self.embedder = None
        self.generator = None

    def init_embedder(self):
        del self.embedder, self.generator, 
        self.embedder = None
        self.generator = None
    
    def albert_embedding(self, sequences):
        self.embedder = embed.ProtTransAlbertBFDEmbedder()
        self.generator = self.embedder.embed_many(sequences)
        features = np.array([np.mean(v, axis=0) for v in self.generator])
        self.init_embedder()
        return features
    
    def bert_embedding(self, sequences):
        self.embedder = embed.ProtTransBertBFDEmbedder()
        self.generator = self.embedder.embed_many(sequences)
        features = np.array([np.mean(v, axis=0) for v in self.generator])
        self.init_embedder()
        return features
    
    def t5_embedding(self, sequences):
        self.embedder = embed.ProtTransT5BFDEmbedder()
        self.generator = self.embedder.embed_many(sequences)
        features = np.array([np.mean(v, axis=0) for v in self.generator])
        self.init_embedder()
        return features
    
    def t5ft_embedding(self, sequences):
        self.embedder = embed.ProtTransT5XLU50Embedder()
        self.generator = self.embedder.embed_many(sequences)
        features = np.array([np.mean(v, axis=0) for v in self.generator])
        self.init_embedder()
        return features
    
    def xlnet_embedding(self, sequences):
        self.embedder = embed.ProtTransXLNetUniRef100Embedder()
        self.generator = self.embedder.embed_many(sequences)
        features = np.array([np.mean(v, axis=0) for v in self.generator])
        self.init_embedder()
        return features