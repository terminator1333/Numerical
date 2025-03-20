"""
In this assignment you should interpolate the given function.
"""


import cProfile
import pstats
class Assignment1:
    def __init__(self):
        """
        Here goes any one time calculation that need to be made before
        starting to interpolate arbitrary functions.
        """
        pass

    def interpolate(self, f: callable, a: float, b: float, n: int) -> callable:

        chebyshev_nodes=np.cos(np.linspace(0, np.pi, n)) * (b - a) / 2 + (a + b) / 2#creating chebyshev nodes, they worked great after testing
        chebyshev_nodes=np.sort(chebyshev_nodes) #sorting the points to use them correctly as intervals later on in the code
        chebyshev_nodes[-1] = b
        values=np.array([f(x) for x in chebyshev_nodes], dtype=np.float64) #storing the f value of each x created in the chebyshev node

        #calculating the barycentric interpolation weights
        weights_bary=np.ones(n)  #initialize all weights_bary to 1
        for first in range(n): #going over each chebyshev node and comparing it with every other chebyshev node, its just the algorithm
            for second in range(n):
                if second==first:  #incase equal, continue to next iteration
                    continue
                weights_bary[first]/=(chebyshev_nodes[first]-chebyshev_nodes[second])  #calculating product of differences
        def interpolating_function(x): #the actual interpolation function that we will retunr
            x = np.asarray(x, dtype=np.float64) #making sure that x is an array
            is_scalar=False #if it is scalar (a number) we need to return .item() in the end, so to remember we do this
            if x.ndim==0:#checking if its a scalar, if so, turning it into an array of size 1
                x =np.array([x])
                is_scalar=True
            exact_match = np.zeros_like(x, dtype=bool)#in order to track if input x exactly matches any Chebyshev node
            numerator=np.zeros_like(x, dtype=np.float64) #initialising numerator and denominator for barycentric interpolation, its just the formula
            denominator=np.zeros_like(x, dtype=np.float64)

            for j in range(n):# computing the sum according to the formula
                diff_bary_x=x-chebyshev_nodes[j]  #computing difference between input x and node j
                mask =diff_bary_x == 0  #finding cases where x matches a node exactly
                exact_match = np.logical_or(exact_match, mask)  #keeping track of exact matches
                w_over_diff = weights_bary[j]/diff_bary_x  # Compute weight contribution for each term
                numerator += values[j]*w_over_diff  #calculating the sum of weighted function values
                denominator += w_over_diff  #calculating the sum  of the weights


            result = np.where(exact_match, values[np.argmax(exact_match, axis=0)], numerator / denominator)#incase x matches a node exactly, we will return the function value to avoid division by zero

            return result.item() if is_scalar else result #incase inserted x was a single number, we will use .item() to return a single number, else we will return an array

        return interpolating_function #finally returning the function

    ##########################################################################


import unittest
from functionUtils import *
class TestAssignment1(unittest.TestCase):

    def test_with_poly(self):
        T = time.time()

        ass1 = Assignment1()
        mean_err = 0

        d = 30
        for i in tqdm(range(100)):
            a = np.random.randn(d)

            f = np.poly1d(a)

            ff = ass1.interpolate(f, -1, 1, 50)

            xs = np.random.random(200)
            err = 0
            for x in xs:
                yy = ff(x)
                y = f(x)
                err += abs(y - yy)

            err = err / 200
            mean_err += err
        mean_err = mean_err / 100

        T = time.time() - T
        print(T)
        print(mean_err)

    def test_with_poly_restrict(self):
        ass1 = Assignment1()
        a = np.random.randn(5)
        def f7(x):
            return pow(np.e, pow(np.e, x))
        f = RESTRICT_INVOCATIONS(10)(f7)
        with cProfile.Profile() as profile:
            ff = ass1.interpolate(f, 1, 2, 10)
            xs = np.random.random(20)
            for x in xs:
                yy = ff(x)
        result =pstats.Stats(profile)
        result.print_stats().sort_stats(pstats.SortKey.TIME)


if __name__ == "__main__":
    unittest.main()
