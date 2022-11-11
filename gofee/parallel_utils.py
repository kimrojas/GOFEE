import numpy as np

def split(Njobs, comm_size):
    """Splits job indices into simmilar sized chuncks
    to be carried out in parallel.
    """
    Njobs_each = Njobs // comm_size * np.ones(comm_size, dtype=int)
    Njobs_each[:Njobs % comm_size] += 1
    Njobs_each = np.cumsum(Njobs_each)
    split = np.split(np.arange(Njobs), Njobs_each[:-1])
    return split

def parallel_function_eval(comm, func, **kwargs):
    """Distributes the results from parallel evaluation of func()
    among all processes.
    
    comm: mpi communicator

    func: function to evaluate. Should return a list of results.
    """
    results = func(**kwargs)
    results_all = comm.gather(results, root=0)
    results_all = comm.bcast(results_all, root=0)
    results_list = []
    for results in results_all:
        if isinstance(results, list):
            results_list += results
        else:
            results_list.append(results)
    return results_list

