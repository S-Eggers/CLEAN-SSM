from argparse import ArgumentParser

argparser = ArgumentParser()

argparser.add_argument("-d", "--datasets", type=str, default="all", help="Datasets to run experiments on (comma separated)")
argparser.add_argument("-e", "--error", type=float, default=0.25, help="Maximum error rate")
argparser.add_argument("-em", "--min_error", type=float, default=0.01, help="Minimum error rate")
argparser.add_argument("-es", "--error_step_size", type=float, default=0.05, help="Step size for error rate")
argparser.add_argument("-g", "--generator", type=int, default=1, help="Error generator, (0) = original garf, (1) = simple, (2) = BART")
argparser.add_argument("-i", "--interval_for_error_removal", type=int, default=0, help="The interval for removing defective tuples from training set, 0 = [0], 1 = [0, 1], 2 = [0, 0.5, 1]")
argparser.add_argument("-l", "--limit", type=int, default=-1, help="Limit the number of dataset rows")
argparser.add_argument("-m", "--method", type=str, default="garf-original", help="Method to run (see README.md 'Experiments available')")
argparser.add_argument("-n", "--n_jobs", type=int, default=1, help="Number of jobs to run in parallel")
argparser.add_argument("-o", "--original_experiment", action="store_true", help="Run the original experiment")
argparser.add_argument("-p", "--prepare", action="store_true", help="Prepare the databases / data lake")
argparser.add_argument("-r", "--runs", type=int, default=5, help="Number of runs per error rate")
argparser.add_argument("-s", "--send", action="store_true", help="Send push notification")
argparser.add_argument("-t", "--tex", action="store_true", help="Generate latex style plots, requires latex installed on system")

args = argparser.parse_args()
