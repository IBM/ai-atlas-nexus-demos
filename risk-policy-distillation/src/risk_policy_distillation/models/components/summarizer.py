import json
import logging
from json import JSONDecodeError

from risk_policy_distillation.models.components.context_generator import (
    ContextGenerator,
)


logger = logging.getLogger("logger")


class Summarizer:

    def __init__(self, inference_engine):
        """
        LLM-based component for summarizing open-text reasoning
        :param llm_component: LLM wrapper component
        """
        self.inference_engine = inference_engine

        cg = ContextGenerator()
        self.summarizing_context = cg.generate_summarization_context()

    def summarize(self, message):
        """
        Summarizes open text reasoning into short bulletpoins
        :param message: Open-text reasoning
        :return:
        """
        messages = [
            {"role": "system", "content": self.summarizing_context},
            {"role": "user", "content": message},
        ]

        output = self.inference_engine.chat([messages])
        bulletpoints = []
        try:
            json_output = json.loads(output[0].prediction)
            bulletpoints = json_output["causes"]

            bulletpoints = [
                b[0].lower() if isinstance(b, list) else b.lower() for b in bulletpoints
            ]
            return bulletpoints
        except JSONDecodeError:
            pass

        return bulletpoints
