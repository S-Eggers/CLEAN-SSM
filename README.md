# CLEAN-SSM: CLEANing data lakes in a Self-Supervised Manner
 
Self-supervised data cleaning experiments. Experiments available:
- GARF (Self-supervised and Interpretable Data Cleaning with Sequence Generative Adversarial Networks)
- RNN-GARF (GARF with LSTM instead of SeqGAN)
- BiLSTM-GARF (GARF with Bidirectional LSTM instead of SeqGAN)
- MPGARF (Multi-Core GARF)
- Tuning-GARF (Hyperparameter search for network size in original GARF)
- Linear-GARF (GARF with FD-detection instead of SeqGAN)
- Distilled-GARF (RNN-GARF with Knowledge Distillation)
- Column-Scalability (How well does GARF scale with increasing column size?)
- Column-Ordering (Does column ordering has any effect on found rules?)
- Tax-GARF (How well does GARF scale with with increasing tuple size?)
- GARF Applicability (What needs to be done to a dataset so that we can not use GARF on it anymore?)
- GARF Enhancement (Can we improve results by duplicating our dataset?)
- Rule-Coverage (How much does a dataset need to change in order that we cannot reusage the found rules?)
- Rule-Reusage (Can we transfer rules to Tax?)
- LakeCleanerNaive (Multiple GARF models, without interaction, single core version)
- LakeCleanerRNNNaive (Multiple RNN-GARF models, without interaction, single core version)
- LakeCleaner (Multiple RNN-GARF models, rule transfer, dataset clustering, single core version)

Run an experiment:
```
main.py [-h] [-d DATASETS] [-e ERROR] [-g GENERATOR]
        [-i INTERVAL_FOR_ERROR_REMOVAL] [-l LIMIT] [-m METHOD]
        [-n N_JOBS] [-o] [-p] [-r RUNS] [-s] [-t]

optional arguments:
  -h, --help            show this help message and exit
  -d DATASETS, --datasets DATASETS
                        Datasets to run on (comma separated)
  -e ERROR, --error ERROR
                        Maximum error rate
  -g GENERATOR, --generator GENERATOR
                        Error generator, (0) = simple, (1) = BART (applies
                        only to original_experiment = False)
  -i INTERVAL_FOR_ERROR_REMOVAL, --interval_for_error_removal INTERVAL_FOR_ERROR_REMOVAL
                        The interval for removing defective tuples from
                        training set, 0 = [0], 1 = [0, 1], 2 = [0, 0.5, 1]
  -l LIMIT, --limit LIMIT
                        Limit the number of dataset rows
  -m METHOD, --method METHOD
                        Method to run
  -n N_JOBS, --n_jobs N_JOBS
                        Number of jobs to run in parallel
  -o, --original_experiment
                        Run the original experiment
  -p, --prepare         Prepare the databases
  -r RUNS, --runs RUNS  Number of runs per error rate
  -s, --send            Send push notification
  -t, --tex             Generate latex style plots, requires latex installed
                        on system
```

## ToDo

- [x] GARF
- [x] RNN-GARF
- [x] BiLSTM-GARF
- [x] MPGARF
- [x] Tuning-GARF
- [x] Linear-GARF
- [x] Distilled-GARF
- [x] GARF Tuple Benchmarking on Tax Dataset w/ different Dataset Sizes (10000, 25000, 50000, 100000, 200000)
- [x] GARF Column Benchmarking on Tax Dataset w/ 10000 Tuples and Column Lengths of 2-15
- [x] GARF Column Ordering Experiment with Rule Difference
- [x] GARF Applicability
- [x] GARF Enhancement
- [x] Rule Coverage
- [x] Rule Reusage
- [x] Uni Detect
- [x] Lake Cleaner Naive
- [x] Lake Cleaner RNN Naive
- [x] Lake Cleaner

## Citations
- Peng, Jinfeng, et al. "Self-supervised and Interpretable Data Cleaning with Sequence Generative Adversarial Networks." Proceedings of the VLDB Endowment 16.3 (2022): 433-446.
- Wang, Pei, and Yeye He. "Uni-detect: A unified approach to automated error detection in tables." Proceedings of the 2019 International Conference on Management of Data. 2019.
