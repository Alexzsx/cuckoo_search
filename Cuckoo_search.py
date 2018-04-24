import numpy as np
import math
from multiprocessing import cpu_count
from pprint import pprint


def get_cuckoos(nest, bestnest, Lb, Ub):
    # Get cuckoos by ramdom walk
    # Levy flight
    beta = 3 / 2.
    sigma = (math.gamma(1 + beta) * math.sin(math.pi * beta / 2) / (
            math.gamma((1 + beta) / 2) * beta * 2 ** ((beta - 1) / 2))) ** (1/beta)

    for j in range(nest.shape[0]):
        single_nest = nest[j, :]
        u = np.random.randn(single_nest.shape[0]) * sigma
        v = np.random.randn(single_nest.shape[0])
        step = u / abs(v) ** (1 / beta)
        step_size = 0.01 * step * (single_nest - bestnest)
        single_nest = single_nest + step_size * np.random.randn(single_nest.shape[0])
        single_nest = single_nest[0]
        nest[j, :] = simplebounds(single_nest, Lb, Ub)

    return nest


def empty_nests(nest, Lb, Ub, pa):
    # Replace some nests by constructing new solutions/nests
    # A fraction of worse nests are discovered with a probability pa
    n = nest.shape[0]
    K = np.random.rand(nest.shape[0], nest.shape[1]) > pa

    nestn1 = nest[np.random.permutation(n), :]
    nestn2 = nest[np.random.permutation(n), :]
    # New solution by biased/selective random walks
    stepsize = np.random.rand() * (nestn1 - nestn2)
    new_nest = nest + stepsize * K

    for i in range(new_nest.shape[0]):
        s = new_nest[i, :]
        new_nest[i, :] = simplebounds(s, Lb, Ub)

    return new_nest


def get_best_nest_cs(nest, new_nest, fitness, Y, Phi, U):
    for i in range(nest.shape[0]):
        f_new = get_fitness(new_nest[i, :], Y, Phi, U)
        if f_new <= fitness[i]:
            fitness[i] = f_new
            nest[i, :] = new_nest[i, :]
    f_min = np.min(fitness)
    index = np.where(fitness == f_min)[0]
    best_nest = nest[index, :]

    return f_min, best_nest, nest, fitness


def get_best_nest(nest, new_nest, fitness):
    for i in range(nest.shape[0]):
        f_new = get_fitness(new_nest[i, :])
        if f_new <= fitness[i]:
            fitness[i] = f_new
            nest[i, :] = new_nest[i, :]
    f_min = np.min(fitness)
    index = np.where(fitness == f_min)[0]
    best_nest = nest[index, :]

    return f_min, best_nest, nest, fitness


def get_fitness_cs(nest, Y, Phi, U):
    D_hat = U[:, nest]
    W = np.linalg.pinv(Phi.dot(D_hat)).dot(Y)
    res = Y - Phi.dot(D_hat).dot(W)
    res_T = res.T
    fitness = res_T.dot(res)

    return fitness


def get_fitness(nest):
    return (pow(nest[0], 2) + pow(nest[1] + 1, 2)) + 4


def simplebounds(single_nest, Lb, Ub):
    dimension = single_nest.shape[0]
    for j in range(dimension):
        if single_nest[j] < Lb[j]:
            single_nest[j] = Lb[j]
        elif single_nest[j] > Ub[j]:
            single_nest[j] = Ub[j]

    return single_nest


if __name__ == '__main__':
    # parameters set
    iteration = 20
    n = 20  # number of nest
    pa = 0.25  # A fraction of worse nests are discovered with a probability pa
    dim = 2  # number of cuckoo in a nest
    Lb = np.ones(dim) * np.array([-10, -10])  # low bounds of cuckoo
    Ub = np.ones(dim) * np.array([10, 10])  # high bounds of cuckoo

    nest = np.zeros((n, dim))
    for i in range(n):
        nest[i, :] = Lb + (Ub - Lb) * np.random.rand(Lb.shape[0])

    fitness = np.ones(n)
    fmin, bestnest, nest, fitness = get_best_nest(nest, nest, fitness)

    for i in range(iteration):
        new_nest = get_cuckoos(nest, bestnest, Lb, Ub)
        _, _, nest, fitness = get_best_nest(nest, new_nest, fitness)
        new_nest = empty_nests(nest, Lb, Ub, pa)
        # find best nest
        fnew, best, nest, fitness = get_best_nest(nest, new_nest, fitness)

        if fnew < fmin:
            fmin = fnew
            bestnest = best

    print bestnest, fmin
