import os

import litellm
import torch
from dotenv import load_dotenv

from risk_policy_distillation.fm_factual.utils import (
    DEFAULT_PROMPT_BEGIN,
    DEFAULT_PROMPT_END,
    HF_MODELS,
    RITS_MODELS,
)

# Import inference engines from ai-atlas-nexus
from ai_atlas_nexus.blocks.inference import (
    InferenceEngine,
)

from ai_atlas_nexus.metadata_base import InferenceEngineType

# Load environment variables
load_dotenv()

GPU = torch.cuda.is_available()
DEVICE = GPU * "cuda" + (not GPU) * "cpu"


class dotdict(dict):
    """dot.notation access to dictionary attributes"""

    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__


class LLMHandler:
    def __init__(self, inference_engine: InferenceEngine, model: str, verbose: bool = False, **default_kwargs):
        """
        Initializes the LLM handler.
        :param inference_engine: InferenceEngine.
        :param model: Model name or path.
        :param verbose: Verbose
        :param default_kwargs: Default parameters (e.g., temperature, max_tokens) to pass to completion calls.
        """
        self.engine = inference_engine
        self.verbose = verbose

        if self.engine._inference_engine_type is InferenceEngineType.RITS:
            self.rits_model_info = RITS_MODELS[model]
            self.prompt_template = self.rits_model_info.get("prompt_template", None)

    def completion(self, prompt, **kwargs):
        """
        Generate a response using the RITS API (if RITS=True) or the local model.

        :param message: The prompt.
        :param kwargs: Additional parameters for completion (overrides defaults).
        """
        return self._call_model(prompt, **kwargs)

    def batch_completion(self, prompts, **kwargs):
        """
        Generate responses in batch using the RITS API (if RITS=True) or the local model.

        :param prompts: List of prompts.
        :param kwargs: Additional parameters for batch completion.
        """
        return self._call_model(prompts, **kwargs)

    def _call_model(self, prompts, num_retries=5, **kwargs):
        """
        Handles both single and batch generation.

        :param prompts: A single string or a list of strings.
        :param kwargs: Additional parameters.
        """
       
        result = self.engine.chat(prompts, verbose=self.verbose)
        return result

