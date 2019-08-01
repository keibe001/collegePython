import matplotlib.pyplot as plt
import numpy as np
import random
import Queue as Q

filepath = "python\iris.txt"
f = open(filepath,"r")
max_iter = 1
a = input("Which problem would you like to demo? (1, 2, 3, or 4) ")
if a == 2 or a == 3:
    max_iter = input("What would you like the maximum iterations to be? ")
i_names = ['sepal-length', 'sepal-width', 'petal-length', 'petal-width', 'class']
i_class = [(0,49),(50,99),(100,149)]
i_class_names = ['Iris-setosa','Iris-versicolor','Iris-virginica']
length = len(i_names)-1       #remove the class form the iris dataset



f1 = f.readlines()
h = len(f1)
sumOfErrorsk3 = []
data = np.zeros((h,len(i_names)-1))
x_index = 0
temp = 0
for x in f1:
    line = x.split(',')
    temp = len(line)-1          #remove the class name from the iris data set, since data is type float and you can't cast a str to float
    for y in range(temp):
        data[x_index][y] = line[y]
    x_index+= 1

def distance(x,y):
    p = 2    
    total = 0
    for i in range(len(x)):                 #summing
        total += abs(x[i] - y[i])**p
    total = total**(1/float(p))             #root of p
    return total

# Question 1
def initialize(x_input,K):                              #randomly initialize the centroids
    cluster_centroids = np.zeros((K,len(x_input[0][:])))
    nums = random.sample(range(149),K)
    for i in range(K):
        cluster_centroids[i][:] = data[nums[i]][:]
    return cluster_centroids

def k_means_cs171(x_input,K,init_centroids): 
    cluster_assignments = np.zeros((len(x_input)))
    cluster_assignments.flatten()
    cluster_centroids = init_centroids
    exiter = False
    allAssignments = []
    allAssignments.append(np.copy(cluster_assignments))
    errorList = []
    while(True):
        sumOfErrors = 0
        for i in range(0,150):                  #assigns points to the nearest cluster
            assignDist = 9999
            for j in range(K):                  #calculating the distance between each point and each centroid
                temp = distance(cluster_centroids[j][:], x_input[i][:])
                if temp < assignDist:
                    assignDist = temp
                    cluster_assignments[i] = int(j)     #assigns the lowest distance to cluster_assignements
            sumOfErrors += assignDist**2
        errorList.append(sumOfErrors)
        for i in range(len(allAssignments)):        #checks if this cluster_assignment has been already calculated
            exiter = True
            for j in range(len(cluster_assignments)):
                if (cluster_assignments[j] != allAssignments[i][j]): #checks if this cluster_Assignment has been calculated
                    exiter = False # it hasn't
                    break
            if exiter:               #this cluster_assignment has already been computed, 
                ret = [cluster_assignments,cluster_centroids]
                kneeRet = [min(errorList),ret]
                if K==3:
                    global sumOfErrorsk3                        # this is if you have multiple iterations
                    sumOfErrorsk3.append(min(errorList))        #this is to print k=3 for question 11
                return kneeRet              #RETURN

        allAssignments.append(np.copy(cluster_assignments))
        for i in range(K):                              #calculating cluster_Centroids
            for f in range(len(i_names) - 1):           #iterate through each feature
                total = []
                for j in range(len(cluster_assignments)): #adds each feature to total
                    if cluster_assignments[j] == i:
                        total.append(x_input[j][f])
                if len(total) > 0:
                    cluster_centroids[i][f] = sum(total) / float(len(total))



# Question 3
def kmeansapp(x_input,K):
    cluster_centroids = np.zeros((K,len(x_input[0][:])))
    num1 = random.randint(0,149)
    cluster_centroids[0][:] = data[num1][:] #initiaizing the first random centroid
    for h in range(1,K):                #filling out the centroid table
        dist = []
        for i in range(len(x_input)): #all remaining data points, you don't have to worry about already selected points since they'll have a min dist of 0
            temp = 9999
            for j in range(h):
               temp2 = distance(cluster_centroids[j][:], x_input[i][:]) #calculating distance between x_input[i] and the closest already selected centroid
               if temp > temp2: 
                   temp = temp2
            temp = temp **2
            dist.append(temp)
        num2 = random.uniform(0,sum(dist))            #how the next random point is selected
        total = 0                                   #farther points are weighted more since they increase total more
        for i in range(len(x_input)):
            total += dist[i]
            if total >= num2:
                cluster_centroids[h][:] = data[i][:] #next centroid selected
                break
    return cluster_centroids


# Question 4
def topData(x_input,K,cluster_assignments, cluster_centroids):
    for j in range(K):
        top3 = Q.PriorityQueue()            #keeps the least distant from the centroids indexes
        for i in range(3):
            top3.put(([-9999,-1])) #uses negative because PriorityQueue.get() returns the lowest
        for i in range(len(cluster_assignments)):
            if int(cluster_assignments[i]) == j: #only lets in clusters assigned to cluster j
                temp = -1 * distance(x_input[i][:], cluster_centroids[int(cluster_assignments[i])][:]) # distance between the cluster and datapoint
                temp2 = top3.get()
                if temp > temp2[0]: # -10 > -9999 #improvement
                    top3.put(([temp,i]))
                else:
                    top3.put(([temp2[0],temp2[1]])) #put back
        output = 'For cluster ' + str(j) + ':'
        for z in range(3):
            temp = top3.get()
            if temp[1] < 50:
                output = output + ' ' + i_class_names[0] #iris-setosa
            elif temp[1] <  100:
                output = output + ' ' + i_class_names[1] #iris-versicolor
            else:
                output = output + ' ' + i_class_names[2] #iris-virginica
        print(output)
    

knee_x = np.arange(1,11)
knee_y = []
means = []
stddev = []
if a == 1 or a==2 or a==3:
    for K in range(1,11):
        sens = []
        for s in range(max_iter):
            if a == 2 or a == 1:
                cent = initialize(data,K)  #completely random
            if a == 3:
                cent = kmeansapp(data,K)    #randomized clusters
            dubbs = k_means_cs171(data,K,cent)
            if s == 0:
                knee_y.append(dubbs[0]) #number 1
                
            sens.append(dubbs[0])       #number 2
        means.append(np.mean(sens))     #number 2
        stddev.append(np.std(sens))     #number 2
    print("Sum of Squared Errors with (k=3) = "  + str(np.mean(sumOfErrorsk3))) 

if a==1:
    plt.plot(knee_x,knee_y)
    plt.title("Knee")
    plt.xticks(knee_x)
    plt.xlabel("Number of Clusters")
    plt.ylabel("Sum of Squared Errors")
    plt.show()
elif a == 2:
    plt.errorbar(knee_x,means,yerr=stddev,capsize = 2,capthick = 5)
    plt.title("Randomly Generated Error bars with " + str(max_iter) + " iterations")
    plt.xticks(knee_x)
    plt.xlabel("Number of Clusters")
    plt.ylabel("Sum of Squared Errors")
    plt.show()
elif a == 3:
    plt.errorbar(knee_x,means,yerr=stddev,capsize = 2,capthick = 5)
    plt.title("K-Means++: Error bars with " + str(max_iter) + " iterations")
    plt.xticks(knee_x)
    plt.xlabel("Number of Clusters")
    plt.ylabel("Sum of Squared Errors")
    plt.show()
elif a == 4:
    K = 3
    cent = kmeansapp(data,K) 
    dubbs = k_means_cs171(data,K,cent)
    topData(data,K,dubbs[1][0],dubbs[1][1])
