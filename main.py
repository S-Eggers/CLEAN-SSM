from multiprocessing import set_start_method
from factory import Factory
from argparser import args
import logging
# from send_push import send
# import sys
# import os
# sys.path.append(os.path.join(os.getcwd(), "methods", "uni_detect_fatemeh"))

if __name__ == '__main__':    
    logging.info("Starting experiments")
    set_start_method('spawn')
    
    # get arguments
    method = args.method
    datasets = args.datasets
    runs_per_error_rate = args.runs
    max_error_rate = args.error
    min_error_rate = args.min_error
    error_step_size = args.error_step_size
    interval_for_error_removal = args.interval_for_error_removal
    limit_rows = args.limit
    n_parallel_jobs = args.n_jobs
    error_generator = args.generator
    
    # currently not supported with BART
    if error_generator == 2:
        max_error_rate = 0.1
    
    # create factory
    factory = Factory(method, datasets, error_generator)
    if args.prepare:
        factory.prepare(limit_rows)
    
    # run experiments
    last_suite = None
    for suite in factory:
        logging.info("Starting experiment suite...")
        
        suite = suite.run(
            max_error_rate,
            min_error_rate,
            error_step_size,
            runs_per_error_rate, 
            interval_for_error_removal,
            n_parallel_jobs=n_parallel_jobs
        )
        # merge suites or store first suite
        if last_suite is None:
            last_suite = suite
        else:
            last_suite.merge(suite)

        logging.info("Finished experiment suite.")
    # plot and save results
    else:
        last_suite.plot().save(args.__dict__)
    
    # send push notification
    #if args.send:
    #    logging.info("Sending push notification...")
    #    send()