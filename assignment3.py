"""
In this assignment you should find the area enclosed between the two given functions.
The rightmost and the leftmost x values for the integration are the rightmost and
the leftmost intersection points of the two functions.

The functions for the numeric answers are specified in MOODLE.


This assignment is more complicated than Assignment1 and Assignment2 because:
    1. You should work with float32 precision only (in all calculations) and minimize the floating point errors.
    2. You have the freedom to choose how to calculate the area between the two functions.
    3. The functions may intersect multiple times. Here is an example:
        https://www.wolframalpha.com/input/?i=area+between+the+curves+y%3D1-2x%5E2%2Bx%5E3+and+y%3Dx
    4. Some of the functions are hard to integrate accurately.
       You should explain why in one of the theoretical questions in MOODLE.

"""
import cProfile
import pstats
import assignment2


class Assignment3:
    def __init__(self):
        """
        Here goes any one time calculation that need to be made before
        solving the assignment for specific functions.
        """

        pass

    import numpy as np
    def integrate(self, f: callable, a: float, b: float, n: int) -> np.float32:
        a, b = np.float32(a),np.float32(b) #turning to type float32
        scale = np.float32(0.5*(b-a))#the scaling factor for transforming the quadrature interval
        #the gauss legendre data for small n (1 to 4) in a dictionary, matching each number of allowed invocations with allowed rule (midpoint, cubic, etc..)
        gauss_legendre_data={1: (np.array([0.0], dtype=np.float32),  np.array([2.0], dtype=np.float32)),2: (np.array([-0.5773502691896, 0.5773502691896], dtype=np.float32),
                np.array([1.0, 1.0], dtype=np.float32)), 3: (np.array([-0.7745966692414, 0.0, 0.7745966692414], dtype=np.float32),np.array([0.5555555555555, 0.8888888888888, 0.5555555555555], dtype=np.float32)),
            4: (np.array([-0.861136311594053, -0.339981043584856, 0.339981043584856, 0.861136311594053], dtype=np.float32),np.array([0.347854845137454, 0.652145154862546, 0.652145154862546, 0.347854845137454], dtype=np.float32))
        }
        if n in gauss_legendre_data: #incase less than 4 calls allowed, check how many exactlt and use gauss legendre accordignly
            points, weights = gauss_legendre_data[n] #retrieving precomputed points and weights for the given n
            transformed_points = scale * points + np.float32(0.5 * (a + b)) #transforming points from the original range from -1 to 1 to the needed range a,b with this formula
            return scale * np.dot(weights, f(transformed_points))#calcualting the integral with the formula: scale*(weights*f(points)) where the * between weights and f points is the dot profuct, so we use it
        kronrod_points = np.array([
            -0.991455371120813, -0.949107912342759, -0.86486442335977,
            -0.741531185599394, -0.586087235467691, -0.40584515137739,
            -0.207784955007899, 0.0, 0.207784955007899, 0.40584515137739,
            0.58608723546769, 0.741531185599394, 0.86486442335977,
            0.9491079123427, 0.9914553711208
        ], dtype=np.float32)  #15 precomputed points on [-1, 1], symmetric around 0, these are the kronrod points
        kronrod_weights = np.array([
            0.0229353220105, 0.06309209262998, 0.10479001032225,
            0.1406532597155, 0.1690047266392, 0.1903505780647,
            0.2044329400752, 0.2094821410847, 0.2044329400752,
            0.1903505780647, 0.1690047266392, 0.1406532597155,
            0.10479001032225, 0.06309209262998, 0.0229353220105
        ], dtype=np.float32)  #the kronrod weights

        first_val=f((float)(a+b)/2) # i know i have at least 2 calls cause else we wouldve finished the function earlier
        second_val=f((float)(a))
        if n >= 15 and a>0 and abs(first_val-second_val)>1e+6: #incase special type like the hard oscilatory test, use this instead
                num_subintervals=n-3 #we already made 2 calls, so max n-3 calls left
                bounds = np.logspace(np.log10(a), np.log10(b), num_subintervals + 1, dtype=np.float32) #using logspace, we can cause a>0 and b>a
                total_integral = np.float32(0.0)#the total calculated area
                for i in range(num_subintervals): #going through all intervals and calculating in them the kronrod points and total area
                    sub_a,sub_b=bounds[i],bounds[i + 1] #the interval
                    sub_scale = 0.5 * (sub_b - sub_a) #half the length of interval, for formula
                    transformed_points =sub_scale*kronrod_points+0.5*(sub_a+sub_b) #the kronrod formula for the transformed points
                    f_evals=f(transformed_points.astype(np.float32)) #the values from the points
                    total_integral += sub_scale * np.sum(kronrod_weights * f_evals) #adding to total area calculated
                return total_integral #returnning the total area
        transformed_points = scale * kronrod_points + np.float32(0.5 * (a + b))#turning kronrod points to the range [a, b] using the same linear mapping as gauss legendre which is the formula scale * kronrod_points + np.float32(0.5 * (a + b))
        return scale * np.dot(kronrod_weights,f(transformed_points))#calcualting the integral with the formula: scale*(weights*f(points)) where the * between weights and f points is the dot profuct, so we use it

    def areabetween(self, f1: callable, f2: callable) -> np.float32:
        assign = assignment2.Assignment2() #using assignment2 to find the intersections
        f1_cach = np.vectorize(f1, otypes=[np.float32]) #caching both f1 and f2 using the vectorize function in python, it works ok
        f2_cach = np.vectorize(f2, otypes=[np.float32])
        cach_minus = {}
        def minus_abs(x):
            x_key = tuple(x) if isinstance(x, np.ndarray) else x #turning x into a tuple incase its a np.array, and if it isnt we keep the x as the same type
            if x_key in cach_minus: #if the x is in the cache, return it
                return cach_minus[x_key]
            result = np.abs(f1_cach(x) - f2_cach(x)) #else, compute the result and store in the cache, and return it
            cach_minus[x_key] = result
            return result
        intersections = assign.intersections(f1, f2, 1, 100, 1) #using assignment2 to find all roots
        f1_at_1, f2_at_1 = f1_cach(1), f2_cach(1) #the f values at f1 and f100, to compare
        f1_at_100, f2_at_100 = f1_cach(100), f2_cach(100)
        if np.isclose(f1_at_1, f2_at_1): #incase for either x=1 or x=100, the f values are very close, we add them as intersection points
            intersections.append(1)
        if np.isclose(f1_at_100, f2_at_100):
            intersections.append(100)
        intersections = sorted(set(intersections)) #using set and then sorting in order to traverse through the intervals as neede

        if len(intersections) < 2: #incase less than 2 return nan as needed
            return np.float32(np.nan)
        total_area = 0.0 #the total area calculated
        for i in range(len(intersections)-1): #going through all intervals found and calculating the area for each one
            a, b=intersections[i], intersections[i+1] #the interval
            area=self.integrate(minus_abs,a,b,15) #15 points because after testing its enough
            total_area+= area  # adding the area calculated to total area
        return np.float32(total_area) #returning the final calculated area

    ##########################################################################


import unittest
from sampleFunctions import *
from tqdm import tqdm


class TestAssignment3(unittest.TestCase):

    def test_integrate_float32(self):
        def f3(x):
            return np.sin(pow(x, 2))

        ass3 = Assignment3()
        areasin=ass3.integrate(f3,0,1,9)
        print("Area Sin: ",areasin)
        print("First--------------------")

        f1 = lambda x: x ** 2 - 50 * x + 600
        f2 = lambda x: 0.5 * x + 10
        area1 = ass3.areabetween(f1, f2)
        print(area1)
        assert not np.isnan(area1) and area1 > 0, "Test 1 Failed: Area should be positive and not NaN"
        print("Second--------------------")
        f1 = lambda x: 30 + 10 * np.sin(0.1 * x)
        f2 = lambda x: 30 - 0.1 * x
        area2 = ass3.areabetween(f1, f2)
        print(area2)
        assert not np.isnan(area2) and area2 > 0, "Test 2 Failed: Area should be positive and not NaN"
        print("Third--------------------")
        f1 = lambda x: 10 * np.log10(x)
        f2 = lambda x: 2 * np.sqrt(x)
        area3 = ass3.areabetween(f1, f2)
        print(area3)
        assert not np.isnan(area3) and area3 > 0, "Test 3 Failed: Area should be positive and not NaN"
        print("Fourth--------------------")
        f1 = lambda x: 0.01 * x ** 3 - x + 20
        f2 = lambda x: -0.02 * x ** 2 + x + 15
        area4 = ass3.areabetween(f1, f2)
        print(area4)
        assert not np.isnan(area4) and area4 > 0, "Test 4 Failed: Area should be positive and not NaN"
        print("Fifth--------------------")
        f1 = lambda x: 50 * np.sin(0.5 * x) + x
        f2 = lambda x: 50 * np.cos(0.5 * x) + x - 10
        area1 = ass3.areabetween(f1, f2)
        print(area1)
        assert not np.isnan(area1) and area1 > 0, "Test 1 Failed!"

        print("Sixth--------------------")
        f1 = lambda x: 0.01 * x ** 3 - 0.5 * x ** 2 + 5 * x
        f2 = lambda x: 20 * np.sin(0.1 * x) + 0.02 * x ** 2
        area2 = ass3.areabetween(f1, f2)
        print(area2)
        assert not np.isnan(area2) and area2 > 0, "Test 2 Failed!"

        print("Seventh--------------------")
        f1 = lambda x: 5 * np.exp(0.03 * x) + 10
        f2 = lambda x: 15 * np.log10(x + 1) + 5
        area3 = ass3.areabetween(f1, f2)
        print(area3)
        assert not np.isnan(area3) and area3 > 0, "Test 3 Failed!"

        print("Eighth--------------------")
        f1 = lambda x: 20 * np.sin(0.2 * x) + 10 * np.sin(0.5 * x) + x / 2
        f2 = lambda x: 15 * np.cos(0.3 * x) + 5 * np.sin(0.7 * x) + x / 2 - 5
        prof = cProfile.Profile()
        prof.enable()
        area4 = ass3.areabetween(f1, f2)
        prof.disable()
        stats=pstats.Stats(prof)
        stats.print_stats()
        print(area4)
        assert not np.isnan(area4) and area4 > 0, "Test 4 Failed!"

        print("All hard tests passed! 🚀")

        print("All tests passed!")

    def test_integrate_hard_case(self):
        ass3 = Assignment3()
        f1 = strong_oscilations()

        # Create a cProfile profiler instance
        profiler = cProfile.Profile()

        # Enable profiling for the function call
        profiler.enable()

        # Perform the integration
        r = ass3.integrate(f1, 0.09, 10, 20)

        # Disable profiling after the function call
        profiler.disable()

        # Print the profiling results
        profiler.print_stats()

        # Check the result
        true_result = -7.78662 * 10 ** 33
        print(r)
        self.assertGreaterEqual(0.001, abs((r - true_result) / true_result))


if __name__ == "__main__":
    unittest.main()
