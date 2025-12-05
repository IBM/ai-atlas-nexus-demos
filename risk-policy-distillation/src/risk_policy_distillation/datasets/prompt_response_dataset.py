from risk_policy_distillation.datasets.abs_dataset import AbstractDataset


class PromptResponseDataset(AbstractDataset):

    def __init__(self, config, dataframe=None):
        super().__init__(config=config, dataframe=dataframe)

    def process_config(self, c):
        super().process_config(c)

        self.response_col = c.get("response_col", "")
        self.expl_input = self.response_col

    def extract_message(self, row):
        prompt = row[self.prompt_col]
        response = row[self.response_col]
        local_expl_input = response

        # get true label
        true_label = row[self.label_col]

        return (prompt, response), local_expl_input, true_label

    def build_message_format(self, message, decision):
        assert len(message) == 2

        prompt, response = message
        return f""" Message:
                        User prompt: {prompt}
                        AI response: {response}
                    Decision: The message is {decision}"""
