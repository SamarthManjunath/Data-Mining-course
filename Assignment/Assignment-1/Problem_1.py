#Name: Samarth Manjunath
#UTA ID: 1001522809

#libraries required to run the program
import numpy
import copy
import matplotlib.pyplot as plt
import itertools

#The mean and the co variance matrix is initialized to their respective values
m1 = [1,0]
s1 = [[0.9,0.4],[0.4,0.9]]
m2 = [0,1.5]
s2 = [[0.9,0.4],[0.4,0.9]]
#using the multivariate function to generate the 2D gaussian random variables
sd1 = numpy.random.multivariate_normal(m1, s1, 500)#first 2D gaussian random number
sd2 = numpy.random.multivariate_normal(m2, s2, 500)#second 2D gaussian random number
rd = numpy.concatenate((sd1,sd2))#merged list of two 2D gaussian random data
k = eval(input("Please enter the number of cltrs into which the clustering needs to be done."))#clutsering input taken here
c = [] #creating a list to keep track of centroids.
for x in range(k):
    vc = input("Please enter the cluster"+ str(x+1) +" : ")#the initial ctrs has to be given here.
    ctrs = []
    for x in vc.split(" "):
        ctrs.append(float(x))
    c.append(ctrs)
c = numpy.asarray(c, dtype = numpy.float64)

#k means method where the clustering takes place based on the input provided by the user.
def myKmeans(X,k,c):
    ld = rd.shape[0]#ld contains the number of data points
    od_ctrs = numpy.zeros(c.shape)#array which contains 0's    
    nw_ctrs = copy.deepcopy(c)
    cltrs = numpy.zeros(ld)
    dtce = numpy.zeros((ld,k))
    stp_cd = numpy.linalg.norm(nw_ctrs-od_ctrs) #stopping condition as such the requirements
    z = 0
    while stp_cd > 0.001:#condition has to be satisified to end the program
        z = z+1
        print("number of iterations:",z)
        if z == 10000:#number of iterations which has to be reached
            break
        #calculating the eucledian distance
        for x in range(k):
            dtce[:, x] = numpy.linalg.norm(rd - nw_ctrs[x], axis = 1)
        cltrs = numpy.argmin(dtce, axis = 1)#assigning the points to the nearest centroid which has been given as input.
        print(cltrs)
        od_ctrs = copy.deepcopy(nw_ctrs)#changing the old centroid values
        #computing the center points by computing the mu values for the points of the cluster.
        for x in range(k):
            nw_ctrs[x] = numpy.mean(rd[cltrs == x], axis = 0 )
        stp_cd = numpy.linalg.norm(nw_ctrs - od_ctrs)#re-compute the stopping critieria here just to make sure it is right.
        print(stp_cd)
    print("Center values post kmeans implementation", nw_ctrs)

    #Scatter plot implementation
    xplts = []
    yplts = []
    for x in rd:
        for d in x:
            if d == x[0]:
                xplts.append(d)
            if d == x[1]:
                yplts.append(d)

    #Here the coloring of the plot is taken care of.
    for p,x,y in zip(cltrs, xplts, yplts):
        if p == 0:
            plt.scatter(x, y, color = 'red')
            continue
        elif p == 1:
            plt.scatter(x, y, color = 'blue')
            continue
        elif p == 2:
            plt.scatter(x, y, color = 'yellow')
            continue
        elif p == 3:
            plt.scatter(x, y, color = 'green')
            continue

    for x,y in nw_ctrs:
        plt.scatter(x, y, color = 'black')


    plt.xlabel("xplts")
    plt.ylabel("yplts")
    plt.show()

#k means method is passed
myKmeans(rd,k,c)

