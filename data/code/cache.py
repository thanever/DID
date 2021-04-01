from collections import defaultdict
import numpy as np
import psutil
import ray
import cvxpy as cp
import time

num_cpus = psutil.cpu_count(logical=True)
ray.init(num_cpus=num_cpus)
@ray.remote
class Model:
    
    def __init__(self):
        
        self.seed = {1:1,2:11,3:5,4:1111,5:1,6:11,7:5,8:1111, 9:1,10:11,11:5,12:1111}
        self.generate()

    def generate_single(self, s):
        n = 3
        p = 3
        np.random.seed(self.seed[s])
        C = np.random.randn(n, n)
        A = []
        b = []
        for i in range(p):
            A.append(np.random.randn(n, n))
            b.append(np.random.randn())

        # Define and solve the CVXPY problem.
        # Create a symmetric matrix variable.
        self.X[s] = cp.Variable((n,n), symmetric=True)
        # The operator >> denotes matrix inequality.
        constraints = [self.X[s] >> 0]
        constraints += [
            cp.trace(A[i]@self.X[s]) == b[i] for i in range(p)
        ]
        self.prob[s] = cp.Problem(cp.Minimize(cp.trace(C@self.X[s])),
                        constraints)

    def generate(self):
        self.X  = dict()
        self.prob = dict()
        for s in [1,2,3,4,5,6,7,8,9,10,11,12]:
            self.generate_single(s)

    def solve_single(self, s):
        self.prob[s].solve(solver = 'MOSEK', verbose = False)


    def get_result(self):
        return self


t1 = time.time()

streaming_actors = [Model.remote() for _ in range(num_cpus)]
for s in [1,2,3,4,5,6,7,8,9,10,11,12]:
    streaming_actors[(s-1) % num_cpus].solve_single.remote(s)

result = ray.get(streaming_actors[1].get_result.remote())

# model = Model()
# for s in [1,2,3,4,5,6,7,8,9,10,11,12]:
#     model.solve_single(s)


t2 = time.time()
print(t2 -t1)








# Print result.
# print("The optimal value is", prob.value)
# print("A solution X is")
# print(X.value)













# num_cpus = psutil.cpu_count(logical=True)
# ray.init(num_cpus=num_cpus)

# @ray.remote

# class StreamingPrefixCount(object):

#     def __init__(self):
#         self.prefix_count = defaultdict(int)
#         self.popular_prefixes = set()

#     def add_document(self, document):
#         for word in document:
#             for i in range(1, len(word)):
#                 prefix = word[:i]
#                 self.prefix_count[prefix] += 1
#                 if self.prefix_count[prefix] > 3:
#                     self.popular_prefixes.add(prefix)

#     def get_popular(self):
#         return self.popular_prefixes


# streaming_actors = [StreamingPrefixCount.remote() for _ in range(num_cpus)]

# # Time the code below.
# for i in range(num_cpus * 10):
#     document = [np.random.bytes(20) for _ in range(10000)]
#     streaming_actors[i % num_cpus].add_document.remote(document)

# # Aggregate all of the results.
# results = ray.get([actor.get_popular.remote() for actor in streaming_actors])
# popular_prefixes = set()

# for prefixes in results:
#     popular_prefixes |= prefixes