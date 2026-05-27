from transformers import AutoTokenizer, AutoConfig, AutoModelForCausalLM

from transformers.models.auto.configuration_auto import CONFIG_MAPPING
from transformers.models.auto.modeling_auto import MODEL_FOR_CAUSAL_LM_MAPPING

from .llada.model_llada import LLaDAConfig, LLaDAForCausalLM
from .dream.model_dream import DreamConfig, DreamForCausalLM

CONFIG_MAPPING.register('llada', LLaDAConfig, True)
MODEL_FOR_CAUSAL_LM_MAPPING.register(LLaDAConfig, LLaDAForCausalLM, True)

CONFIG_MAPPING.register('dream', DreamConfig, True)
MODEL_FOR_CAUSAL_LM_MAPPING.register(DreamConfig, DreamForCausalLM, True)