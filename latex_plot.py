import os
from factory import Factory
import matplotlib.pyplot as plt
import scienceplots
import pandas as pd
from search.search import Search
from argparse import ArgumentParser


argparser = ArgumentParser()
argparser.add_argument("-m", "--method", type=str, default="garf-original", help="Method to run")
argparser.add_argument("-i", "--input_dir", type=str, default="results", help="Input directory")
argparser.add_argument("-d", "--dataset", type=str, help="Dataset name")
argparser.add_argument("-s", "--search", action="store_true", help="Use grid search results")
argparser.add_argument("-a", "--aggregate", action="store_true", help="Aggregate and plot results")
args = argparser.parse_args()

if __name__ == '__main__':
    plt.style.use("science")
    if args.search:
        Search.debug_plot(args.input_dir)
    elif args.aggregate:
        directories = args.input_dir.split(",")
        methods = args.method.split(",")
        combined_df = None
        _suite = None
        for directory, method in zip(directories, methods):
            factory = Factory(method, args.dataset)
            csv_path = None
            for suite in factory:
                _suite = suite
                csv_path = suite.set_result_path(directory)
                break
            
            df = pd.read_csv(os.path.join(csv_path, "results.csv"), index_col=0)
            df["method"] = f"{method}-{directory}"
            if combined_df is None:
                combined_df = df
            else:
                combined_df = pd.concat([combined_df, df])
                
        _suite.combined_debug_plot(combined_df)
        combined_df.to_csv("combined.csv")
        
    else:
        factory = Factory(args.method,args.dataset)
        for suite in factory:
            suite.debug_plot(args.input_dir)
