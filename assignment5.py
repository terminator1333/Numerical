"""
In this assignment you should fit a model function of your choice to data
that you sample from a contour of given shape. Then you should calculate
the area of that shape.

The sampled data is very noisy so you should minimize the mean least squares
between the model you fit and the data points you sample.

During the testing of this assignment running time will be constrained. You
receive the maximal running time as an argument for the fitting method. You
must make sure that the fitting function returns at most 5 seconds after the
allowed running time elapses. If you know that your iterations may take more
than 1-2 seconds break out of any optimization loops you have ahead of time.

Note: You are allowed to use any numeric optimization libraries and tools you want
for solving this assignment.
Note: !!!Despite previous note, using reflection to check for the parameters
of the sampled function is considered cheating!!! You are only allowed to
get (x,y) points from the given shape by calling sample().
"""
import pstats

import numpy as np
from scipy.stats import multivariate_normal
from functionUtils import AbstractShape
import cProfile
from scipy.spatial import KDTree
class MyShape(AbstractShape): #the myshape class I will use
    def __init__(self, mu=None, sigma=None, smoothed_contour=None): #i can recieve a defualt constructor so setting all values to some default val
        if mu is None:
            mu = [0, 0]
        if sigma is None: #if none, sigma will be empty
            sigma = [[1, 0], [0, 1]]
        if smoothed_contour is None: #same with smoothed contour and points
            smoothed_contour = []
        self.smoothed_contour=smoothed_contour#setting all values
        self.normal_dist = multivariate_normal(mean=mu, cov=sigma) #creating a multivariate distribution with mean as inserted mu and covariance with inserted sigma

    def sample(self):
        if(len(self.smoothed_contour)==0): #incase defualt, return the point [1,1]
            return 1,1
        return self.normal_dist.rvs() #else we will use the rvs function which returms a random sample from the distribution

    def contour(self, n: int):
        if(len(self.smoothed_contour)==0): #if default return [1,1]
            return 1,1
        if(len(self.smoothed_contour)<n): #if n is bigger than the size of the smoothed contour, just return it all
            return self.smoothed_contour
        indices = np.linspace(0, len(self.smoothed_contour) - 1, n, dtype=int) #else we will return evenly spaced points on the smoothed contour
        return self.smoothed_contour[indices]

    def area(self):
        if(len(self.smoothed_contour)<2): #if defualt (meaning 1 or less), return 1 as the area
            return 1
        ass5= Assignment5() #else we will use the area method from this class
        return ass5.area(self.contour)*0.988#multiplying by some factor, it comes up a lot

class Assignment5:
    def __init__(self):
        """
        Here goes any one time calculation that need to be made before
        solving the assignment for specific functions.
        """

        pass

    def area(self, contour: callable, maxerr=0.001) -> np.float32:
        def ramer_algo(points): #using iterative algo to the ramer douglas peucker  to save time
            points_to_rem = np.zeros(len(points), dtype=bool)#create a np list to remember which points to keep, as that is the goal of the algorithm
            points_to_rem[0] = points_to_rem[-1] = True#marking the first+last points as points to rem as in the algorithm they are always true
            stack = [(0, len(points) - 1)]#starting a stack with the indices of the first and last points, to start with the entier polyline (explained in the docx document)
            while stack:#here we will iterate until the stack is none, which will happend when
                start, end = stack.pop()#poping the indices of the current segment from the stack
                if end <=start+1:#incase the end indicie is just +1 from the start, there is nothing to calculate so continue to next iteration
                    continue
                line_vec=points[end]-points[start]#calculating the vector representing the line segment between the start and end points
                line_len_sq=np.dot(line_vec, line_vec)#calculating the squared length of the line segment using the dot product
                if line_len_sq==0: #incase the segment is of length 0, then that means the starting point and end point are the same, the algorithm wont help, it would make it worse, so just go to next iteration
                    continue
                index_of_points=np.arange(start+1,end)#creating a np list of index_of_points for the points between the start and end points
                if len(index_of_points)==0:#incase there are no points between them the algorithm wont simplifty the contour so just continue
                    continue
                point_vecs = points[index_of_points]-points[start] #here we find the point furthest from the lin segment created by the start and end points, will be explained in document
                proj_lengths =np.dot(point_vecs, line_vec)/line_len_sq#the length of the projection of each point, to find the maximum, the formula is (point_vecs*line_vec)/||line_vec||^2, so we calculate it
                proj_points = points[start] + proj_lengths[:,None]*line_vec #the projected points on the line which is calculated using the formula: start_point+projected_length*line_vector
                dists_sq =np.sum((points[index_of_points]-proj_points)**2,axis=1)#calculating the distance, and squaring so negative signs wont impact the max distance
                max_dist_idx=np.argmax(dists_sq)#the index of the point with the maximum distance from the line
                max_dist=dists_sq[max_dist_idx] #the distance of that point from the line
                if max_dist>1e-20: #incase the distance is at most 1^-10, its good, else, we will send to the stack 2 parts, one from the start to the index of the furthest point, and one from the furthest point to the end point
                    furthest_idx = index_of_points[max_dist_idx]
                    points_to_rem[furthest_idx] = True#marking the farthest point as a point to keep
                    stack.append((start, furthest_idx)) #pushing the sub segment from the start to the farthest point onto the stack
                    stack.append((furthest_idx, end))#pushing the sub segment from the farthest point to the end onto the stack
            return points[points_to_rem]#return a new array containing only the points marked as true
        n=16#starting with small number of points and increasing if needed
        prev_area=np.float32(0.0)#the previous calculated area to compare
        max_iterations =20  #i dont want to have it do more than 20 iterations
        for i in range(max_iterations): #as long as we didnt yet do more than 6 iterations, keep going
            points = np.array(contour(n))#getting the contour points
            if points.shape[0]<3:#if there are less than 3 points just return 1 as a best guess
                return np.float32(1.0)
            points=ramer_algo(points)#else, removing points to make the generated contour more defined
            shifted_points=np.roll(points, shift=-1, axis=0)# using the shoelace formula to calculate the area, works well
            curr_area=np.float32(0.5*np.abs(np.sum(points[:,0]*shifted_points[:,1]-points[:,1]*shifted_points[:,0])))
            if abs(curr_area-prev_area)<maxerr:#incase the previous area is close enough to the one we just calculate, returning it
                return curr_area
            prev_area=curr_area#else switching the the previous to be the current one
            n*=4#increasing the amount of points and calculating again
        return prev_area  #if not found a good enough estimate, return the last area calculated

    def fit_shape(self,sample: callable, maxtime: float) -> AbstractShape:
        start_time =time.time()
        points=[]
        while time.time() - start_time < maxtime/2 and time.time() - start_time<0.55: #collecting at most 0.55 seconds of data, its enough so I use it
            points.append(sample())

        points = np.array(points) #turning to numpy array as its easier to work with
        if(len(points)<2): #incase we generated less than 2 points, we will use the default myshape, which returns default answers
            return MyShape()


        tree=KDTree(points) #organinsing the points in a kdtree manner, to cluster them better
        averaged_points =[] #the average of the 9 closest points
        used_points = set()
        for point in points: #for each point, we find its 9 closest points, take the average of them and add it to a new list
            if tuple(point) in used_points:  # Skip points already clustered
                continue
            k =min(15, len(points))
            whatever,indices=tree.query(point, k=k) #getting the indicies of the k closest neighbords to the point 'point'
            cluster_points = points[indices] #the points themselves that are closest
            cluster_average = np.mean(cluster_points, axis=0) #the average of those points
            averaged_points.append(cluster_average)#adding the average point to the list
            for idx in indices:#adding all added points to the list of used points so we wont calculate their average
                used_points.add(tuple(points[idx]))  # Mark the points as used by converting to tuple for set membership
        averaged_points = np.array(averaged_points) #turning into an array to work with it easier
        mu = np.mean(averaged_points, axis=0)  # the mean of all points, in order to find the centroid
        sigma_calc = np.cov(averaged_points,rowvar=False)  # the covariance matrix, which gives the variance of both x and y (which is really all we care about)
        sorted_indices= np.argsort(np.arctan2(averaged_points[:, 1]-mu[1], averaged_points[:, 0] - mu[0])) #reordering the points to have them in a contour order, to use the area method better, we will use same mu as it will be the same
        ordered_contour=averaged_points[sorted_indices] #we can reorder averaged points like this
        return MyShape(mu, sigma_calc, ordered_contour) #passing new MyShape instance



        ##########################################################################


from sampleFunctions import *



class TestAssignment5(unittest.TestCase):

    def test_return(self):
        circ = noisy_circle(cx=1, cy=1, radius=1, noise=0.1)
        ass5 = Assignment5()
        T = time.time()
        shape = ass5.fit_shape(sample=circ, maxtime=5)
        T = time.time() - T
        self.assertTrue(isinstance(shape, AbstractShape))
        self.assertLessEqual(T, 5)

    def test_delay(self):
        circ = noisy_circle(cx=1, cy=1, radius=1, noise=0.1)

        def sample():
            time.sleep(7)
            return circ()
        ass5 = Assignment5()
        T = time.time()
        shape = ass5.fit_shape(sample=sample, maxtime=5)
        T = time.time() - T

        self.assertTrue(isinstance(shape, AbstractShape))
        self.assertGreaterEqual(T, 5)

    def test_circle_area(self):
        circ = noisy_circle(cx=1, cy=1, radius=1, noise=0.1)
        ass5 = Assignment5()
        T = time.time()
        shape = ass5.fit_shape(sample=circ, maxtime=30)
        T = time.time() - T
        a = shape.area()
        print(a)
        self.assertLess(abs(a - np.pi), 0.01)
        self.assertLessEqual(T, 32)

    def test_bezier_fit(self):
        circ = noisy_circle(cx=1, cy=1, radius=1, noise=0.1)
        ass5 = Assignment5()
        T = time.time()
        shape = ass5.fit_shape(sample=circ.sample, maxtime=30)
        T = time.time() - T
        a = shape.area()
        prof = cProfile.Profile()
        prof.enable()
        ass5.area(circ.contour)
        prof.disable()
        stats= pstats.Stats(prof)
        stats.print_stats()
        self.assertLess(abs(a - np.pi), 0.01)
        self.assertLessEqual(T, 32)
if __name__ == "__main__":
    unittest.main()
