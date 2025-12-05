import json

from risk_policy_distillation.models.components.context_generator import (
    ContextGenerator,
)


class Labeller:

    def __init__(self, inference_engine):
        """
        LLM-based component producing labels for clusters of concepts
        :param llm_component: LLM wrapper component
        """
        self.inference_engine = inference_engine

        cg = ContextGenerator()
        self.labeling_context = cg.generate_labeling_context()

    def label(self, additional_context, cluster, temperature):
        """
        Labels a cluster using LLM
        :param additional_context: a list of previously tried cluster labels
        :param cluster: A list of concepts to be labelled
        :param temperature: Labelling LLM temperature parameter
        :return: A common label for the cluster
        """

        # appending previously used labels to the context to encourage creativity
        context = self.labeling_context + additional_context

        prompt = """
                 Bulletpoints: {bulletpoints}
                 """.format(
            bulletpoints=cluster
        )

        messages = [
            {"role": "system", "content": context},
            {
                "role": "user",
                "content": prompt,
            },
        ]

        # prompting LLM to label the cluster
        cluster_name = self.inference_engine.chat([messages])
        cluster_name = json.loads(cluster_name[0].prediction)["common_reason"]

        return cluster_name
