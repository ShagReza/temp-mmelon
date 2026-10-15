import numpy as np
import torch
import torch.nn as nn
from bmfm_sm.predictive.data_modules.graph_finetune_dataset import Graph2dFinetuneDataPipeline
from bmfm_sm.predictive.data_modules.image_finetune_dataset import ImageFinetuneDataPipeline
from bmfm_sm.predictive.data_modules.text_finetune_dataset import TextFinetuneDataPipeline
from joblib import Parallel, delayed
from torch.nn.functional import pad
from torch.nn.utils.rnn import pad_sequence
from bmfm_sm.api.smmv_api import SmallMoleculeMultiViewModel, PredictionIterator
from bmfm_sm.core.data_modules.namespace import LateFusionStrategy
from bmfm_sm.api.dataset_registry import DatasetRegistry

def compute_embedding(smi, model_path1):
    from bmfm_sm.api.smmv_api import SmallMoleculeMultiViewModel

    emb = SmallMoleculeMultiViewModel.get_embeddings(
        smiles=smi,
        model_path=model_path1,
        huggingface=False,
    )

    if hasattr(emb, "detach"):
        emb = emb.detach().cpu().numpy()

    emb = np.asarray(emb).reshape(-1)
    return smi, emb.tolist()