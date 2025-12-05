import logging
from pathlib import Path

import pandas as pd
from sentence_transformers import SentenceTransformer

from tqdm import tqdm

from risk_policy_distillation.models.components.reasoner import Reasoner
from risk_policy_distillation.models.components.summarizer import Summarizer
from risk_policy_distillation.models.components.verifier import Verifier
from risk_policy_distillation.models.guardians.judge import Judge


logger = logging.getLogger("logger")


class Extractor:

    def __init__(
        self,
        guardian: Judge,
        inference_engine,
        risk_type,
        risk_definition,
        local_explainer=None,
    ):
        """
        Local explanation generation component (analogous to CLoVE algorithm)
        :param guardian: LLM-as-a-Judge being explained
        :param llm_component: wrapper component for querying an LLM
        :param risk_type:
        :param risk_definition:
        :param local_explainer:
        """
        self.risk_type = risk_type
        self.risk_definition = risk_definition
        self.guardian = guardian

        self.reasoner = Reasoner(inference_engine, guardian)
        self.summarizer = Summarizer(inference_engine)
        self.verifier = Verifier(inference_engine)

        self.local_explainer = local_explainer

    def extract_concepts(self, dataset, save_path: Path, use_lime=False, verbose=False):
        try:
            ds = pd.read_csv(save_path, header=0)
            if len(ds) == dataset.size():
                # if all inputs are processed already
                logger.info("Loaded concepts from {}".format(save_path))
            else:
                # start extracting from the last processed index
                last_index = ds.iloc[-1].Index
                last_id = ds[ds.Index == last_index].index.item() + 1
                logger.info(
                    "Loaded concepts from {} inputs. Continuing from index = {}".format(
                        len(ds), last_index
                    )
                )
                return self._extract_concepts(
                    dataset, save_path, use_lime, verbose, start_id=last_id
                )

        except FileNotFoundError:
            return self._extract_concepts(dataset, save_path, use_lime, verbose)

    def _extract_concepts(
        self, dataset, save_path: Path, use_lime=False, verbose=False, start_id=0
    ):
        logger.info("Generating local explanations...")
        for i, row in tqdm(dataset.train[start_id:].iterrows()):
            # generate a message from a dataframe row
            message, lime_input, true_label = dataset.extract_message(row)

            # judge the message using a guardian
            guardian_response = self.guardian.ask_guardian(message)

            # select important words for each decision using a local explainer such as LIME
            words = None
            if use_lime:
                words = self.local_explainer.explain(
                    lime_input, self.guardian.labels, self.guardian.predict_proba
                )

            # extract verified supporting and conflicting arguments
            bulletpoints = {}
            # for each label
            for i, d in enumerate(self.guardian.labels):
                send_message = dataset.build_message_format(
                    message, self.guardian.label_names[d]
                )
                verified_bulletpoints = self.get_verified_bulletpoints(
                    send_message, d, lime_input, words, use_lime, verbose
                )

                bulletpoints[d] = verified_bulletpoints

            # bulletpoints = self.remove_redundancies(bulletpoints)

            # save results in a dataframe
            self.save_results(
                dataset,
                row,
                message,
                guardian_response,
                true_label,
                bulletpoints,
                save_path,
            )

            if verbose:
                logger.info(message)
                logger.info("Guardian response = {}".format(guardian_response))
                if use_lime:
                    logger.info("Lime words = {}".format(words))
                logger.info("Verified bulletpoints: {}".format(bulletpoints))

        logger.info(
            "Explained {} instances. Results saved in {}".format(
                dataset.size(), save_path
            )
        )

    def get_verified_bulletpoints(
        self,
        message,
        decision,
        lime_input,
        important_words=None,
        use_lime=False,
        verbose=False,
    ):
        reasoning = self.reasoner.reason(message)
        if verbose:
            logger.info(reasoning)
        # extract supporting bulletpoints
        bulletpoints = self.summarizer.summarize(reasoning)

        # verify extracted bulletpoints
        if use_lime:
            bulletpoints = self.verifier.verify(
                bulletpoints, lime_input, important_words[decision]
            )

        return bulletpoints

    def remove_redundancies(self, bulletpoints):
        removes = {k: [] for k in bulletpoints.keys()}
        for decision_1, bullets_1 in bulletpoints.items():
            for decision_2, bullets_2 in bulletpoints.items():
                if decision_1 != decision_2:
                    model = SentenceTransformer("all-MiniLM-L6-v2")
                    if not len(bullets_1) or not len(bullets_2):
                        continue

                    emb_1 = model.encode(bullets_1)
                    emb_2 = model.encode(bullets_2)

                    similarities = model.similarity(emb_1, emb_2)
                    remove = similarities > 0.95

                    for i in range(len(bullets_1)):
                        for j in range(len(bullets_2)):
                            if remove[i][j]:
                                removes[decision_1] = bullets_1[i]
                                removes[decision_2] = bullets_2[j]

        for k, bullets in bulletpoints.items():
            for v in bullets:
                if v in removes[k]:
                    bulletpoints[k].remove(v)
        return bulletpoints

    def save_results(
        self,
        dataset,
        row,
        message,
        guardian_response,
        true_label,
        bulletpoints,
        save_path: Path,
    ):
        if isinstance(message, list) or isinstance(message, tuple):
            message_names = [dataset.prompt_col, dataset.response_col]
            records = [
                [
                    row[dataset.index_col],
                    *message,
                    self.risk_type,
                    self.guardian.label_names.index(guardian_response),
                    true_label,
                ]
                + [bulletpoints[d] for d in self.guardian.labels]
            ]
        else:
            message_names = [dataset.prompt_col]
            records = [
                [
                    row[dataset.index_col],
                    message,
                    self.risk_type,
                    self.guardian.label_names.index(guardian_response),
                    true_label,
                ]
                + [bulletpoints[d] for d in self.guardian.labels]
            ]

        # save dataset periodically
        df = pd.DataFrame(
            records,
            columns=["Index"]
            + message_names
            + ["Criterion", "GG Label", "True Label"]
            + self.guardian.label_names,
        )

        if save_path.exists():
            df.to_csv(save_path, mode="a", index=False, header=False)
        else:
            df.to_csv(save_path, mode="w", index=False)
