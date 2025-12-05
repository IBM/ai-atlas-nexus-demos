from risk_policy_distillation.datasets.abs_dataset import AbstractDataset


class PromptDataset(AbstractDataset):

    def __init__(self, config, dataframe=None):
        super().__init__(config, dataframe)

        self.expl_input = self.prompt_col

    def extract_message(self, row):
        prompt = row[self.prompt_col]
        local_expl_input = prompt

        # get true label
        true_label = row[self.label_col]

        return prompt, local_expl_input, true_label

    def build_message_format(self, message, decision):
        return f""" Message:
                        User prompt: {message}
                        Decision: The message is {decision}"""
