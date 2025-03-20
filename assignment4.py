"""
In this assignment you should fit a model function of your choice to data
that you sample from a given function.

The sampled data is very noisy so you should minimize the mean least squares
between the model you fit and the data points you sample.

During the testing of this assignment running time will be constrained. You
receive the maximal running time as an argument for the fitting method. You
must make sure that the fitting function returns at most 5 seconds after the
allowed running time elapses. If you take an iterative approach and know that
your iterations may take more than 1-2 seconds break out of any optimization
loops you have ahead of time.

Note: You are NOT allowed to use any numeric optimization libraries and tools
for solving this assignment.

"""

class Assignment4:
    def __init__(self):
        """
        Here goes any one time calculation that need to be made before
        solving the assignment for specific functions.
        """

        pass
    def fit(self, f: callable, a: float, b: float, d: int, maxtime: float) -> callable:
        """
        Here we will fit a polynomial to points using chebyshev nodes and polynomial regression

        polynomial regression is very similar to what we learned, so it was fairly easy to implement

        heres a rlly good vid i watched to help understand it :https://www.youtube.com/watch?v=QptI-vDle8Y


        """
        starting_time = time.time()  # tracking time, to know when to exit function
        def find_range_bounds(f, start: float, end: float, start_time=None):
            x_vals = np.arange(start, end, 0.1)
            range_start = None
            range_end = None
            for x in x_vals:
                if time.time()-start_time>maxtime:
                    break
                try:
                    whatever=f(x)
                    if range_start is None:
                        range_start=x
                    range_end=x
                except Exception:
                    continue

            return range_start, range_end


        a, b = find_range_bounds(f, a, b, start_time=starting_time)#finding correct range
        if a is None or b is None or time.time() - starting_time > maxtime:
            return lambda x:1
        def gaussian_elimination(matrix, solved, start_time):#using gaussian elimination to solve matrix
            num_rows = len(matrix)#get the number of rows (or columns since A is square)
            for row in range(num_rows):
                if time.time()-start_time> maxtime:
                    return solved  #returning what we have so far

                diag = matrix[row][row]#extract the diagonal element for normalization
                if abs(diag) < 1e-8:
                    diag = 1e-8
                for column in range(row, num_rows):
                    matrix[row][column] /= diag#normalize the current row by dividing by the diagonal element
                solved[row] /= diag#normalize the corresponding entry in b
                for curr_row in range(row + 1, num_rows):
                    factor = matrix[curr_row][row]#factor to eliminate the i-th variable from row k
                    for column in range(row, num_rows):#subtract the scaled i-th row from row k
                        matrix[curr_row][column]-=factor*matrix[row][column]
                    solved[curr_row]-=factor*solved[row]

            final_solution = [0]*num_rows#initialize the solution vector with zeros
            for row in range(num_rows-1,-1,-1):
                total = 0#initialize the sum for substitution
                for col in range(row+1,num_rows):
                    total += matrix[row][col] * final_solution[col]#sum the known terms
                final_solution[row] = solved[row]-total#calculate the value for the i-th variable
            return final_solution

        def polynomial_regression(data_points_x, data_points_y, polynomial_degree,start_time):  # fitting a polynomial to data points using regression
            def vandermonde_matrix(data_points_x, polynomial_degree): #creating a vandermonde matrix as taught in class
                x=np.array(data_points_x, dtype=np.float32) #turning the datapoints into a numpy array for easier use
                d=polynomial_degree+1  #the number of columns we will need which is the degree of the polynomial+1
                n=len(x)  #the number of rows
                V=np.zeros((n, d), dtype=np.float32)#initiating the vander matrix with n rows and d columns
                for j in range(d):#filling each column with x raised to power j where j goes from 0 to d-1
                    V[:,j]=x**j  #computing x^j for all x vals
                return V#finally, returning the created matrix
            vandermonde_matrix =vandermonde_matrix(data_points_x, polynomial_degree)#generating the vandermonde matrix
            vandermonde_transpose = vandermonde_matrix.T#getting the traspose of the matrix
            matrix_transpose_times_matrix = np.dot(vandermonde_transpose, vandermonde_matrix)# calculating the product of the transpose and the original matrix
            y_column_vector = []# convert the y values into a column vector format for matrix multiplication
            for single_y_value in data_points_y:
                y_column_vector.append([single_y_value])  #appending each y value as a single-element list
            y_column_vector = np.array(y_column_vector)  #converting the list of lists into a numpy array


            matrix_transpose_times_y_vector = np.dot(vandermonde_transpose, y_column_vector)# calculating the product of the transpose of the vandermonde matrix and the y column vector

            y_vector = []
            for row in matrix_transpose_times_y_vector:
                y_vector.append(row[0])
            polynomial_coefficients = gaussian_elimination(matrix_transpose_times_matrix,y_vector,start_time)


            return polynomial_coefficients #finally, returning the computed polynomial coefficients

        def chebyshev_nodes(a, b, n): #creating chebyshev nodes, similar to previous assignments
            return 0.5 * (a+b + (b - a) * np.cos((2 * np.arange(n) + 1) * np.pi / (2 * n)))


        num_of_nodes = int(32770 * np.log2(d+2)) #creating nodes to create the polynomial with
        x_vals = chebyshev_nodes(a, b, num_of_nodes)  #getting the x vals from chebyshev

        f_vals = np.vectorize(f)(x_vals)
        coeffs = polynomial_regression(x_vals, f_vals, d, starting_time) #getting the polynomial coefficients with time check


        def returned_function(x):#creating the returned function using the numpy polyval method
            return np.polyval(coeffs[::-1], x)

        return returned_function


##########################################################################


import unittest
from sampleFunctions import *
from tqdm import tqdm


class TestAssignment4(unittest.TestCase):

    def test_return(self):
        f = NOISY(0.01)(poly(1, 1, 1))
        ass4 = Assignment4()
        T = time.time()
        shape = ass4.fit(f=f, a=0, b=1, d=10, maxtime=5)
        T = time.time() - T
        self.assertLessEqual(T, 5)


    def test_delay(self):
        @NOISY(3)
        def f2_noise(x):
            return pow(x, 2) - 3 * x + 2
        delay=0.005
        f = DELAYED(delay)(f2_noise)

        ass4 = Assignment4()
        T = time.time()
        shape = ass4.fit(f=f, a=0, b=5, d=2, maxtime=5)
        T = time.time() - T
        self.assertGreaterEqual(T, 5)

    def test_err(self):
        f = poly(1, 1, 1)
        nf = NOISY(1)(f)
        ass4 = Assignment4()
        T = time.time()
        ff = ass4.fit(f=nf, a=0, b=1, d=10, maxtime=5)
        T = time.time() - T
        mse = 0
        for x in np.linspace(0, 1, 1000):
            self.assertNotEquals(f(x), nf(x))
            mse += (f(x) - ff(x)) ** 2
        mse = mse / 1000
        print(mse)


if __name__ == "__main__":
    unittest.main()
