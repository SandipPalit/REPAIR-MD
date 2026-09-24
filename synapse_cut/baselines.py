from .llada_official import generate as reference_llada_generate
from .decoder import ConfidenceDecoder,EntropyDecoder,RandomDecoder,AttentionCentralityDecoder,SynapseCutDecoder

def build_decoder(method,model,tokenizer,mask_id,device,generation_positions=None):
    m=method.lower()
    cls={'confidence':ConfidenceDecoder,'mdlm':ConfidenceDecoder,'standard':ConfidenceDecoder,'entropy':EntropyDecoder,'random':RandomDecoder,'attention':AttentionCentralityDecoder,'attention_centrality':AttentionCentralityDecoder}.get(m,SynapseCutDecoder if m in {'synapse_cut','synapse','greedy','beam','interaction','local'} else None)
    if cls is None:raise ValueError(f'Unknown local method: {method}')
    return cls(model,tokenizer,mask_id,device,generation_positions=generation_positions)
