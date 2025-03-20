"""
In this assignment you should find the intersection points for two functions.
"""
import cProfile
import pstats
from io import StringIO


class Assignment2:
    def __init__(self):
        pass

    from typing import Callable, Iterable

    def intersections(self, f1: Callable, f2: Callable, a: float, b: float, maxerr=0.001) -> Iterable:ne
        try:
            func_cach = {}  # initialising the cache
            def f(x):  # creating the subtraction function as when y=0, f1(x) = f2(x), this way is much easier to find the points
                if( type(x) is np.float32):
                    x=float(x)
                if x not in func_cach:  # I will use cache, found online that it rlly helps runtime
                    func_cach[x] = f1(x) - f2(x)
                return func_cach[x]
            def bisection(interval):
                a = interval[0]  #extracting the range
                b = interval[1]
                iterations = 0  # don't want the function to work forever so in order to limit the amount of iterations we can use some index
                while abs(b - a) > maxerr and iterations < 10:  # while the error is still big, keep making the distrance from 0 smaller using bisection method. also if the number of iterations is more than 10 this likely means we are not finding a good interval
                    iterations+=1  # increasing the number of iterations made
                    mid =(a+b)/2  # getting the middle of the range, as should be done
                    f_mid_result=f(mid)  # getting the result of f(mid) and storing in variable to reduce the runtime
                    if abs(f_mid_result)<maxerr:  # incase mid is enough , return it
                        return mid #incase the mid is close enough to the point, its fine wth me so return it
                    if f(a)*f_mid_result<0:  # in case mid is not enough, find the next mid
                        b=mid  # we need to insert mid in b in this case, according to algorithm
                    else:
                        a= mid  # else a needs to be mid
                return (a+b)/2  # finally, if not found a good enough mid, return the best guess which is the next mid
            def mullers_algo(a, b, c):
                orig_a = a
                orig_c =c
                for x in range(10):  # i want to limit the iterations for this algo
                    try:
                        f_a_result=f(a)  # lets initially get the y values to save the runtime
                        f_b_result=f(b)
                        f_c_result=f(c)

                        if abs(c-b)<1e-12 or abs(b-a)<1e-12:#calculating the slopes with some precaution
                            if(f_b_result<1e-11):
                                return b
                            if(f_c_result<1e-11):
                                return c
                            else:
                                return a
                        slope1 = (f_b_result - f_a_result) / (b-a)  # this is the slope (mean) between the points a and b
                        slope2 = (f_c_result - f_b_result) / (c-b)  # this is the slope but between points b and c
                        # we need to slove the quadratic formula that makes the parabola that fits a b and c
                        # in order to do that we need to calculate the coefficients of the parabola, this is how it's done according to the algorithm:
                        a_coeff=(slope2 - slope1) / (c-a)  # the ax^2 coefficient
                        b_coeff= a_coeff * (c-b)+slope2  # the bx coefficient
                        c_coeff=f_c_result  # the c coefficnet = f_c
                        discriminant=(b_coeff**2)-4*a_coeff*c_coeff  #when solving a quadratic formula, we need to first find the discriminant
                        if discriminant <0 or abs(a_coeff)<1e-11:  # we cannot accepnt negative discriminants, as this will result in complex roots, so return none signaling failure, also if the first one is close to 0 just return none as it is problematic
                                return bisection([orig_a,orig_c])
                        sqrt_discriminant = np.sqrt(discriminant)  # getting the sqrt of the discriminant, as done in the formula
                        x1 = c-(2*c_coeff)/(b_coeff+sqrt_discriminant)  # this is the first ans out of the discriminant
                        x2 = c-(2*c_coeff)/(b_coeff-sqrt_discriminant)  # this is the second
                        better_root = x2 if abs(x1 - b) > abs(x2 - b) else x1
                        if abs(f(better_root)) < 1e-11:  # incase the root is like rlly close to 0, just return it, to not waste runtime and refrain from any divide by 0 errors
                            return better_root
                        a=b  # else, do another iteration, by rotating the values, as described in the algorithm
                        b=c
                        c=better_root

                    except ValueError: # if there was a problem, just return b
                        return b
                return b
            n_intervals = int((b - a) * 1000)
            chebyshev_nodes = 0.5 * (a + b) + 0.5 * (b - a) * np.cos((2 * (n_intervals - np.arange(n_intervals)) + 1) / (2 * n_intervals) * np.pi) #creating the chebyshev nodes
            intervals = [[chebyshev_nodes[i], chebyshev_nodes[i - 1]] for i in range(1, len(chebyshev_nodes)) if f(chebyshev_nodes[i - 1]) * f(chebyshev_nodes[i]) < 0] #storing the created intervals from the chebyshev nodes s.t the intervals have opposing signs
            if not intervals: #if no opposing sign intervals were found, we return an empty list to show that these functions have no interecption points
                return []
            results=[] #else we will store the found roots here
            for interval in intervals:
                if f(interval[0]) * f(interval[1]) < 0: #going through each found interval and using bisection on it
                    root = bisection(interval)
                    results.append(root) #adding the found root to the results
            if not results: #just checking again, to be sure
                return []
            delta = (b-a)/(10*n_intervals) #we will need a small delta to use mullers algo so here it is
            for i, root in enumerate(results): #refining each found root using mullers algo
                a,b,c = root-delta,root,root+delta #setting the 3 points as needed
                refined_root = mullers_algo(a,b,c)
                if refined_root is not None: #if we got a number, insert it instead of the bisection ans
                    results[i] = refined_root

            final_results = [root for root in results if abs(f(root)) < maxerr] #removing all points that are more than maxerr, as needed
            return final_results if final_results else [] #finally returning the final results

        except:
                return [0]





##########################################################################


import unittest
from sampleFunctions import *
from tqdm import tqdm


class TestAssignment2(unittest.TestCase):

    def test_sqr(self):

        ass2 = Assignment2()

        f1 = lambda x: 10 * np.log(x)
        f2 = lambda x: 2 * np.sqrt(x)
        X = ass2.intersections(f1, f2, 1, 100,maxerr=0.001)
        print(X)
        for x in X:
            self.assertGreaterEqual(0.001, abs(f1(x) - f2(x)))

    def test_poly(self):
        # Create an instance of Assignment2
        ass2 = Assignment2()

        # Generate the random intersecting polynomials
        f1, f2 = randomIntersectingPolynomials(10)

        # Start profiling
        pr = cProfile.Profile()
        pr.enable()

        # Call the intersections method and get the results
        X = ass2.intersections(f1, f2, -1, 1, maxerr=0.001)

        # Get the roots of the equation f1 - f2 = 0
        roots = np.roots(f1 - f2)
        roots = roots[np.isreal(roots)].real

        # Check that all the roots satisfy the condition f1(x) ≈ f2(x)
        for x in X:
            self.assertGreaterEqual(0.001, abs(f1(x) - f2(x)))

        # Stop profiling
        pr.disable()

        # Create a stream to capture the profiling output
        s = StringIO()
        ps = pstats.Stats(pr, stream=s)
        ps.sort_stats('cumtime')  # Sort by cumulative time
        ps.print_stats()  # Print the profiling result

        # Output the profiling results
        print(s.getvalue())


if __name__ == "__main__":
    unittest.main()
