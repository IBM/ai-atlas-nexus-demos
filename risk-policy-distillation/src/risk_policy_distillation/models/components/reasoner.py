from risk_policy_distillation.models.components.context_generator import (
    ContextGenerator,
)
from risk_policy_distillation.models.guardians.judge import Judge


class Reasoner:

    def __init__(self, inference_engine, guardian: Judge):
        """
        LLM-based component for generating open-text reasoning about a decision
        :param inference_engine: LLM wrapper component
        :param guardian: LLM-as-a-Judge being explained
        """
        self.inference_engine = inference_engine

        cg = ContextGenerator()
        self.reasoning_context = cg.generate_reasoning_context(
            guardian.task, guardian.definition, guardian.label_names
        )

    def reason(self, message) -> str:
        """
        Produces LLM-generated open-text reasoning on a message.
        :param message: A message containing LLM-as-a-Judge input and its decision
        :return: Open-text reasoning about the decision
        """
        messages = [
            {"role": "system", "content": self.reasoning_context},
            {"role": "user", "content": message},
        ]
        reasoning = self.inference_engine.chat([messages])

        return reasoning[0].prediction
