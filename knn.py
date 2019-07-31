import matplotlib.pyplot as plt
import numpy as np
import sys

filepath = "python\cancer.txt"
names = ['Sample code number','Clump Thickness','Uniformity of Cell Size','Uniformity of Cell Shape',
'Marginal Adhesion','Single Epithelial Cell Size','Bare Nuclei','Bland Chromatin','Normal Nucleoli',
'Mitoses','Class']
global x_matrix
global y_vector
global knn_results

def distance(x,y,p):
    total = 0
    for i in range(len(x)):                 #summing
        total += abs(x[i] - y[i])**p
    total = total**(1/float(p))             #root of p
    return total

def knn_classifier(x_test, x_train, y_train, k, p):
    y_test = []
    
    for i in range(len(x_test)):                                    #iterates through x_test
        neighbors = []
        for j in range(len(x_train)):                               #compares x_test[i] to every data point in x_train[j]
            neighbors.append([distance(x_test[i],x_train[j],p),j])      #puts every result into the list neighbors, alongside it's index j
        neighbors.sort()                                            #sorts the list, putting the least distant point on top
        labels = []
        for j in range(k):                                          #k closest neighbors
            temp = y_train[neighbors[j][1]]                         #temp = the j-th closest neighbor
            contains = False
            for z in labels:                                        #labels contains the [counts,label]
                if temp == z[1]:
                    z[0] += 1                                       #increments the count if the label is already present
                    contains = True
            if not contains:
                labels.append([1,y_train[neighbors[j][1]]])         #otherwise, append it to labels
        labels.sort(reverse= True)                                  #sorts so the highest count is last
        y_test.append(labels[0][1])
    return y_test

def question1(p):
    cutoff = 559
    x_train = x_matrix[0:cutoff,:]                          #partitioning the data        
    x_test = x_matrix[cutoff:,:]
    y_train = y_vector[0:cutoff]
    y_test = y_vector[cutoff:]
    
    acc_count = 0.0
    tp_count = 0.0
    tn_count = 0.0
    pos_count = 0.0
    neg_count = 0.0
    k = 1                                   # single closest neighbor
    knn_test = knn_classifier(x_test, x_train, y_train, k, p)
    
    for t in range(len(y_test)):
        if y_test[t] == knn_test[t]:
            acc_count+= 1
        if y_test[t] == 4:        #Acutal Postitive
            pos_count += 1
            if knn_test[t] == 4:    #True Positive
                tp_count += 1
        if y_test[t] == 2:        #Acutal Negative
            neg_count += 1
            if knn_test[t] == 2:    #True Negative
                tn_count += 1
    print("Question 1: k-Nearest Neighbor Classifier with 80/20 split") 
    print("Accuracy: "+ str(acc_count/float(len(y_test))))
    print( "Sensitivity: " + str(tp_count/float(pos_count)) )
    print("Specificity: " + str(tn_count/float(neg_count)) + "\n")



def crossValidation(k,p):
    accuracy = []
    sens = []
    spec = []
    ret = []
    partition = len(data) / float(10)
    for x in range(10):
        lower_bound = int(x*round(partition))
        upper_bound = int((x+1)*round(partition))
        x_train = np.delete(x_matrix,slice(lower_bound,upper_bound ),0)         #partitioning the data
        x_test = x_matrix[lower_bound:upper_bound,:]
        y_train = np.delete(y_vector,slice(lower_bound, upper_bound ),0)
        y_test = y_vector[lower_bound:upper_bound]
        knn_test = knn_classifier(x_test, x_train, y_train, k, p)               #function call to knn_classifier
        acc_count = 0.0
        tp_count = 0.0
        tn_count = 0.0
        pos_count = 0.0
        neg_count = 0.0
        for t in range(len(y_test)):                                        #measuring the performance
            if y_test[t] == knn_test[t]:
                acc_count+= 1
            if y_test[t] == 4:        #Acutal Postitive
                pos_count += 1
                if knn_test[t] == 4:    #True Positive
                    tp_count += 1
            if y_test[t] == 2:        #Acutal Negative
                neg_count += 1
                if knn_test[t] == 2:    #True Negative
                    tn_count += 1
        accuracy.append(acc_count/float(len(y_test)))                           #performance for each fold of data
        sens.append(tp_count/float(pos_count))
        spec.append(tn_count/float(neg_count))
    ret = [   [np.mean(accuracy),np.std(accuracy)],[np.mean(sens),np.std(sens)]  ,[np.mean(spec),np.std(spec)]   ]
    knn_results.append(ret)

def helper(metric,k,p):                                                     #checks if this k/p have been already computed
    if len(knn_results) <= ((p*10) +k):
        crossValidation(k+1,p+1)                                            # if it hasn't it's passed into crossValidation
    return knn_results[(p*10) +k][metric]                                   #returns the tuple [mean,std.dev] for the metric
    
    
def grapher():
    titles = ["Accuracy","Sensitivity","Specificity"]
    k_axis = np.arange(1,11)
    plt.figure(1)
    optimal = np.zeros(20)
    for x in range(3):                                             #For each of the performance metrics, 
        temp = []
        if x == 0:                                                  #creating the subplots
            plt.subplot(221)
        elif x == 1:
            plt.subplot(222)
        else:
            plt.subplot(223)
        for p in range(2):                                          #p = 1 or 2
            performances = []
            deviations = []
            best_k = [[0,0],[0,0]]                                  
            for k in range(10):                                     #k = 1:10
                temp = helper(x,k,p)                                    # gets the [mean,std.dev] for x metric
                optimal[(p*10)+k] += temp[0]                        #measurement of performance, sum of means for all metrics
                if temp[0] > best_k[p][0]:                          #measurement of performance, records the best k for each metric/p value
                    best_k[p][0] = temp[0]
                    best_k[p][1] = k +1                                     #1-10
                performances.append(temp[0])
                deviations.append(temp[1])
            if p == 0:                                                                          #results for p = 1
                one = plt.errorbar(k_axis,performances,yerr=deviations,c="r",capsize = 2,capthick = 5)
                print(titles[x] + " (p=" + str(p+1) + ") had an average performance of " + str(np.mean(performances)) + " and performed best with k = " + str(best_k[p][1]) )
            else:                                                                               #results for p = 2
                two =  plt.errorbar(k_axis,performances,yerr=deviations,c="b",capsize = 2,capthick = 5)
                print(titles[x] + " (p=" + str(p+1) + ") had an average performance of " + str(np.mean(performances)) + " and performed best with k = " + str(best_k[p][1]) )
        plt.title(titles[x])
        plt.legend((one,two),("p = 1", "p = 2"),scatterpoints=1,loc='lower left',ncol=1,fontsize=8)
        plt.xlabel("k neighbors")
        plt.ylabel("Performance")
        plt.xticks(k_axis)
    if sum(optimal[:9]) > sum(optimal[10:]):                        #Determines best p
        print("Best p is 1")
    else:
        print("Best p is 2")
    best_neighbor = [0,0]                                           #measuring the best p
    for x in range(len(optimal)/2):                                 
        temp = optimal[x] + optimal[x+10]
        if best_neighbor[0] < temp:
            best_neighbor[0] = temp
            best_neighbor[1] = x+1
    print("Best k is " + str(best_neighbor[1]) )
    best_neighbor = [0,0]                                           #resets back to 0
    for x in range(len(optimal)):                                   #measures best pair
        if best_neighbor[0] < optimal[x]:
            best_neighbor[0] = optimal[x]
            best_neighbor[1] = x
    print("The best combo was p = " + str(int(best_neighbor[1]/10)+1) + " k = " + str((best_neighbor[1]%10)+1))
    plt.show()
    
#main
f = open(filepath,"r")
f1 = f.readlines()
data = np.zeros((699,10))
avg = []
x_index = 0
for x in f1:
    line = x.split(',')
    for y in range(len(line)):
        if y == 6 and line[y] != '?':
            avg.append(float(line[y]))
        if y != 0:                          #remove the sample code number
            if line[y] != '?':
                temp = float(line[y])
            else:
                temp = 0
            data[x_index][y-1] = temp
            
    x_index += 1

average = np.mean(avg)
for x in range(len(data)):
    if data[x][5] == 0:
        data[x][5] = average
        #replacing the ? with the average. using 5 since the question marks are only in the 5th index
        
        
        
data2 = np.copy(data)     
randomize = raw_input("Would you like to randomize the data? (y/n) ")
if randomize == "y":
    np.random.shuffle(data2)
elif randomize != "n":
    print("Defaulted to not randomize")

#Global variables
x_matrix = np.delete(data2,9,1)
y_vector = data2[:,9]
knn_results = []

quest = input("Which problem would you like to demo? (1 or 2) ")
if quest == 1:
    p_query = input("What would you like p to be? (1 or 2) ")
    if p_query ==1 or p_query == 2:
        question1(p_query)
    else:
        print("Error with p input")
elif(quest == 2):
    grapher()
else:
    print("error with input")
    
