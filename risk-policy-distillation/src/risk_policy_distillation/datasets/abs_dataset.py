import logging
import numpy as np

logger = logging.getLogger('logger')


class AbstractDataset:
    # TODO: allow config to have default parameters

    def __init__(self, dataframe=None, config=None):
        self.process_config(config)

        self.dataframe = self.preprocess_df(dataframe)
        self.train = self.dataframe

    def process_config(self, config):
        # TODO: verify all columns are passed and exist in the dataframe
        if config is None:
            return

        if not len(config):
            return

        self.dataset_name = config.get('dataset_name', 'default')

        self.index_col = config.get('data', '')
        self.prompt_col = config.get('prompt_col', '')
        self.response_col = config.get('response_col', '')
        self.label_col = config.get('label_col', '')

    def size(self):
        if self.dataframe is None:
            return 0

        return len(self.dataframe)

    def preprocess_df(self, dataframe):
        if self.index_col == '':
            dataframe['Index'] = np.arange(0, len(dataframe))
            dataframe = dataframe.rename(columns={self.index_col: 'Index'})
            self.index_col = 'Index'

        if self.label_col == '':
            dataframe['label'] = [1] * len(dataframe)
            dataframe = dataframe.rename(columns={self.label_col: 'label'})
            self.label_col = 'label'

        return dataframe