import logging
from create_db import dataset_names
from insert_error import ORIGINAL, DETECTION


class Factory:
    def __init__(self, approach: str, datasets: str, error_generator: int = 0):
        self.approach = approach.lower()
        self.error_generator = error_generator
        if approach == "tax-garf":
            self.datasets = ["Tax"]
        elif approach in ["lake-cleaner", "lake-cleaner-naive", "lake-cleaner-rnn-naive"]:
            self.datasets = ["data-gov-sandbox"]
        elif approach == "column-scalability":
            self.datasets = ["Tax"]
        elif datasets == "all":
            self.datasets = list(dataset_names.keys())
            if "Tax" in self.datasets:
                self.datasets.remove("Tax")
        else:
            self.datasets = datasets.split(",")
    
    def prepare(self, limit: int = -1, **kwargs):
        if self.approach == "uni-detect":
            from preparation.uni_detect import UniDetectPreparation
            logging.info(f"Preparing Uni-Detect for datasets {self.datasets}")
            UniDetectPreparation("uni_detect_fatemeh", self.datasets, limit).run()
            logging.info(f"Finished preparing Uni-Detect for datasets {self.datasets}")
        elif self.approach in ["lake-cleaner", "lake-cleaner-naive", "lake-cleaner-rnn-naive"]:
            from preparation.lake_cleaner import LakeCleanerPreparation
            logging.info(f"Preparing data lake for {self.datasets}")
            method = self.approach.replace("-", "_")
            LakeCleanerPreparation(method, limit=limit).run()
            logging.info(f"Finished preparing data lake for {self.datasets}")
        elif self.approach == "column-scalability":
            logging.info("Preparing GARF database")
            from preparation.column_garf import GARFColumnPreparation 
            GARFColumnPreparation("garf_original", self.datasets[0], self.error_generator, limit).run()
            logging.info("Finished preparing GARF database")
        elif self.approach == "column-ordering":
            logging.info("Not preparing GARF database, happens directly in experiment")
        elif self.approach == "garf-applicability":
            logging.info("Preparing GARF applicability database")
            from preparation.applicability_garf import GARFApplicabilityPreparation
            GARFApplicabilityPreparation("garf_original", self.datasets[0], self.error_generator, limit).run(kwargs["dataset"])
            logging.info("Finished preparing GARF applicability database")
        elif self.approach == "garf-enhancement":
            logging.info("Preparing GARF applicability database")
            from preparation.enhancement_garf import GARFEnhancementPreparation
            GARFEnhancementPreparation("garf_original", self.datasets[0], self.error_generator, limit).run(kwargs["dataset"])
            logging.info("Finished preparing GARF applicability database")
        elif self.approach == "garf-error-types":
            logging.info("Preparing GARF applicability database")
            from preparation.enhancement_garf import GARFEnhancementPreparation
            GARFEnhancementPreparation("garf_original", self.datasets[0], self.error_generator, limit).run(kwargs["dataset"])
            logging.info("Finished preparing GARF applicability database")
        else:
            from preparation.garf import GARFPreparation
            for dataset in self.datasets:
                if self.approach in ["garf-original", "tuning-garf", "rule-reusage", "rule-transfer", "garf-detector"]:
                    logging.info("Preparing GARF database")
                    GARFPreparation("garf_original", dataset, self.error_generator, limit).run()
                    logging.info("Finished preparing GARF database")
                elif self.approach == "mpgarf":
                    logging.info("Preparing MPGARF database")
                    GARFPreparation("mpgarf", dataset, self.error_generator, limit).run()
                    logging.info("Finished preparing MPGARF database")
                elif self.approach in ["rnn-garf", "rnn-garf-detector"]:
                    logging.info("Preparing RNN-GARF database")
                    GARFPreparation("rnn_garf", dataset, self.error_generator, limit).run()
                    logging.info("Finished preparing RNN-GARF database")
                elif self.approach == "rnn-garf-old":
                    logging.info("Preparing RNN-GARF database")
                    GARFPreparation("rnn_garf_old", dataset, self.error_generator, limit).run()
                    logging.info("Finished preparing RNN-GARF database")
                elif self.approach == "linear-garf":
                    logging.info("Preparing Linear-GARF database")
                    GARFPreparation("linear_garf", dataset, self.error_generator, limit).run()
                    logging.info("Finished preparing Linear-GARF database")
                elif self.approach == "distilled-garf":
                    logging.info("Preparing Distilled-GARF database")
                    GARFPreparation("distilled_garf", dataset, self.error_generator, limit).run()
                    logging.info("Finished preparing Distilled-GARF database")
                elif self.approach == "bilstm-garf":
                    logging.info("Preparing BiLSTM-GARF database")
                    GARFPreparation("bilstm_garf", dataset, self.error_generator, limit).run()
                    logging.info("Finished preparing BiLSTM-GARF database")
                elif self.approach == "demo-claening":
                    GARFPreparation("garf_custom", dataset, self.error_generator, limit).run()
                elif self.approach == "transfer-learning-garf":
                    logging.info("Preparing Transfer-Learning-GARF database")
                    GARFPreparation("transfer_learning_garf", dataset, self.error_generator, limit).run()
                    logging.info("Finished preparing Transfer-Learning-GARF database")

    def __iter__(self):
        for dataset in self.datasets:
            print(dataset, self.datasets)
            yield self._get_suite(dataset)
    
    def get_dataset_name(self) -> str:
        return dataset_names[self.dataset]
    
    def _get_suite(self, dataset: str):
        if self.approach == "garf-original":
            from experiments.garf_suite import GARFSuite
            # See experiments/garf_experiment.py
            logging.info("Choose GARF experiment suite")
            return GARFSuite(dataset, error_generator=self.error_generator)
        elif self.approach == "mpgarf":
            from experiments.mpgarf_suite import MPGARFSuite
            # see experiments/mpgarf_experiment.py
            logging.info("Choose MultiProcessGARF (MPGARF) experiment suite")
            return MPGARFSuite(dataset, error_generator=self.error_generator)
        elif self.approach == "rnn-garf":
            from experiments.rnn_garf_suite import RNNGARFSuite
            # See experiments/rnn_garf_experiment.py
            logging.info("Choose RNN-GARF experiment suite")
            return RNNGARFSuite(dataset, error_generator=self.error_generator)
        elif self.approach == "rnn-garf-old":
            from experiments.rnn_garf_old_suite import RNNGARFOldSuite
            # See experiments/rnn_garf_experiment.py
            logging.info("Choose RNN-GARF experiment suite")
            return RNNGARFOldSuite(dataset, error_generator=self.error_generator)
        elif self.approach == "tuning-garf":
            from experiments.hyperparameter_search_garf_suite import HypeparameterSearchGARFSuite
            # See experiments/hyperparameter_search_garf_suite.py
            logging.info("Choose Random Search + GARF experiment suite")
            return HypeparameterSearchGARFSuite(dataset, error_generator=self.error_generator)
        elif self.approach == "linear-garf":
            from experiments.linear_garf_suite import LinearGARFSuite
            # See experiments/linear_garf_experiment.py
            logging.info("Choose Linear-GARF experiment suite")
            return LinearGARFSuite(dataset, error_generator=self.error_generator)
        elif self.approach == "distilled-garf":
            from experiments.distilled_garf_suite import DistilledGARFSuite
            # See experiments/distilled_garf_experiment.py
            logging.info("Choose Distilled-GARF experiment suite")
            return DistilledGARFSuite(dataset, error_generator=self.error_generator)
        elif self.approach == "tax-garf":
            from experiments.tax_garf_suite import TaxGARFSuite
            # See experiments/tax_garf_experiment.py
            return TaxGARFSuite(dataset, error_generator=self.error_generator)
        elif self.approach == "bilstm-garf":
            from experiments.bilstm_garf_suite import BiLSTMGARFSuite
            # See experiments/bilstm_garf_experiment.py
            return BiLSTMGARFSuite(dataset, error_generator=self.error_generator)
        elif self.approach == "transfer-learning-garf":
            from experiments.transfer_learning_garf_suite import TransferLearningGARFSuite
            # See experiments/transfer_learning_garf_experiment.py
            return TransferLearningGARFSuite(dataset, error_generator=self.error_generator)
        elif self.approach == "demo-cleaning":
            from experiments.demo_cleaning_suite import DemoCleaningSuite
            # See experiments/demo_cleaning.py
            return DemoCleaningSuite(dataset, error_generator=self.error_generator)
        elif self.approach == "rule-reusage":
            from experiments.rule_reusage_suite import RuleReusageSuite
            # See experiments/rule_reusage_experiment.py
            return RuleReusageSuite(dataset, error_generator=self.error_generator)
        elif self.approach == "rule-transfer":
            from experiments.rule_transfer_suite import RuleTransferSuite
            # See experiments/rule_transfer_experiment.py
            return RuleTransferSuite(dataset, error_generator=self.error_generator)
        elif self.approach == "uni-detect":
            from experiments.uni_detect_suite import UniDetectSuite
            # See experiments/uni_detect_experiment.py
            logging.info("Choose Uni-Detect experiment suite")
            return UniDetectSuite(dataset, error_generator=self.error_generator)
        elif self.approach == "lake-cleaner":
            from experiments.lake_cleaner_suite import LakeCleanerSuite
            # See experiments/lake_cleaner_experiment.py
            logging.info("Choose Lake-Cleaner experiment suite")
            return LakeCleanerSuite(dataset, error_generator=self.error_generator)
        elif self.approach == "lake-cleaner-naive":
            from experiments.lake_cleaner_naive_suite import LakeCleanerNaiveSuite
            # See experiments/lake_cleaner_naive_experiment.py
            logging.info("Choose Lake-Cleaner-Naive experiment suite")
            return LakeCleanerNaiveSuite(dataset, error_generator=self.error_generator)
        elif self.approach == "lake-cleaner-rnn-naive":
            from experiments.lake_cleaner_rnn_naive_suite import LakeCleanerRNNNaiveSuite
            # See experiments/lake_cleaner_rnn_naive_experiment.py
            logging.info("Choose Lake-Cleaner-RNN-Naive experiment suite")
            return LakeCleanerRNNNaiveSuite(dataset, error_generator=self.error_generator)
        elif self.approach == "uni-detect-debug":
            from experiments.uni_detect_debug_suite import UniDetectDebugSuite
            # See experiments/uni_detect_experiment.py
            logging.info("Choose Uni-Detect-Debug experiment suite")
            return UniDetectDebugSuite("toy", error_generator=self.error_generator)
        elif self.approach == "column-scalability":
            from experiments.column_scalability_suite import ColumnScalabilitySuite
            logging.info("Choose Column-Scalability experiment suite")
            return ColumnScalabilitySuite(dataset, error_generator=self.error_generator)
        elif self.approach == "column-ordering":
            from experiments.column_ordering_suite import ColumnOrderingSuite
            logging.info("Choose Column-Ordering experiment suite")
            return ColumnOrderingSuite(dataset, error_generator=self.error_generator)
        elif self.approach == "garf-applicability":
            from experiments.garf_applicability_suite import GARFApplicabilitySuite
            logging.info("Choose GARF-Applicability experiment suite")
            return GARFApplicabilitySuite(dataset, error_generator=self.error_generator)
        elif self.approach == "garf-enhancement":
            from experiments.garf_enhancement_suite import GARFEnhancementSuite
            logging.info("Choose GARF-Enhancement experiment suite")
            return GARFEnhancementSuite(dataset, error_generator=self.error_generator)
        elif self.approach == "garf-error-types":
            from experiments.garf_error_types_suite import GARFErrorTypeSuite
            logging.info("Choose GARF-Error-Types experiment suite")
            return GARFErrorTypeSuite(dataset, error_generator=self.error_generator)
        elif self.approach == "garf-detector":
            from experiments.garf_detector_suite import GARFDetectorSuite
            logging.info("Choose GARF-Detector experiment suite")
            return GARFDetectorSuite(dataset, error_generator=DETECTION)
        elif self.approach == "rnn-garf-detector":
            from experiments.rnn_garf_detector_suite import RNNGARFDetectorSuite
            logging.info("Choose RNN-GARF-Detector experiment suite")
            return RNNGARFDetectorSuite(dataset, error_generator=DETECTION)
        else:
            raise ValueError("Invalid approach")