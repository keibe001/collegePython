import matplotlib.pyplot as plt
import numpy as np
import sys 

i_names = ['sepal-length', 'sepal-width', 'petal-length', 'petal-width', 'class']
i_class = [(0,49),(50,99),(100,149)]
i_class_names = ['Iris-setosa','Iris-versicolor','Iris-virginica']
w_class_names = ['1','2','3']
w_names = ['class','Alcohol','Malic acid', 'Ash','Alcalinity of ash','Magnesium','Total phenols','Flavanoids',
'Nonflavanoid phenols', 'Proanthocyanins','Color intensity','Hue','OD280/OD315 of diluted wines','Proline']
w_class = [(0,58),(59,129),(130,177)]
class_num = 0
i = True
print('Which problem would you like to do?')
print('"1a" - Histogram\n"1b" - Box Plot')
print('"2a" - Correlation Plot\n"2b" - Scatter Plot\n"2c" - Distance')

prob = raw_input('Enter(1a,1b,2a,2b,2c): ')                                                 # prompt the user for what question they want to execute
if prob != '1a' and prob != '1b' and prob != '2a' and prob != '2b' and prob != '2c':
    print('Error with input, you typed something not accepted. Try again.')
    sys.exit(0)
if prob != '2b':
    print('Which data set would you like?')
    data_set = input('Type 1 for Iris, 2 for Wine: ')
    if data_set == 1:
        i = True
    elif data_set == 2:
        i = False
    else:
        print('Error with data set, defaulted to Iris\n')
        i = True
if i:
    names = i_names
    length = len(names)-1       #remove the class form the iris dataset
    classes = i_class
    class_names = i_class_names
else:
    names = w_names
    length = len(names)
    classes = w_class
    class_names = w_class_names
#names are the attributes
#length is to prevent an error with reading the csv file's string as a float
#classes is to seperate the histograms and distinguish class type in the distance function
#class_names are the names of classes

if prob == '1a' or prob == '1b' or prob == '2b':
    for x in range(length):
        if not i and x!=0:
            print(str(x) + ') ' + names[x])
        if i:
            print(str(x+1) + ') ' + names[x])
    if i:
        index_1 = input('Enter the number of the attribute you want as your axis: ') -1
    else:
        index_1 = input('Enter the number of the attribute you want as your axis: ') #wine doesn't have the minus because of the class is first
    if index_1 < 0 or index_1 > length-1 or (index_1 == 0 and not i):
        print('Error with index, defaulted to 1')
        if i:
            index_1 = 0
        else:
            index_1 = 1
    if prob == '2b':
        index_2 = input('Enter the number of the attribute you want as your other axis: ') - 1
        if index_2 < 0 or index_2 > length-1:
            print('Error with index, defaulted to 2')
            index_1 = 1
    else:
        class_num = input('Which class number would you like (1,2,3): ') - 1
        if class_num <0 or class_num >2:
            print('Error with class number, defaulted to 1.')
            class_num = 0
if prob == '2c':
    p_order = input('What would you like as your p order? ')
    if p_order != 1 and p_order != 2:
        print('Error with p_order, only 1 or 2 accepted. Defaulted to 1')
        p_order = 1
if prob == '1a':
    binNum = input('What binNum would you like? ')
#index_1 is index of the csv (or attribute) you want to plot
#index_2 is for the second axis in the scatterplot question(2b)
#p_order is for the p order in the distance question (2c)
#binNum is the number of bins you want (5,10,50,100) in the histogram question (1a)


#####################################################################################  Question 1
#    1)Histograms
def histogram(data):
    data = data[classes[class_num][0]:classes[class_num][1]]            #cuts out the data from the classes you don't want to graph
    binHeight = []
    y_axis = []
    data_min = float(min(data))
    data_max = float(max(data))
    data_dif = (data_max - data_min) / (binNum)   #determine the range of values for the bins
    borders = []
    for x in range(0,binNum):
        binHeight.append(0)
        y_axis.append(data_min + data_dif*x)               #set the y_axis and initialize the histogram  
        if x != 0:                                          # borders prints the specific dimensions
            borders.append(str(data_min + data_dif*(x-1)) + '-' + str(data_min + data_dif*x))
    print('Histogram bin sizes: ' )
    print(borders)
    
    for x in data:              #fillout the histogram
        if float(x) == data_max:
            x = float(x) - data_dif
            #this is here because the histogram would always use one bin for the max element in the dataset, 
        binHeight[int((float(x)-data_min)/data_dif)] += 1           #increment binHeight for each value that falls within it's range
    y_pos = np.arange(len(y_axis)) 
    plt.bar(y_pos, binHeight, 1, align='edge', alpha=0.5)
    plt.xticks(y_pos,y_axis)
    plt.ylabel('#')
    plt.title(names[index_1] + ' For Class ' + class_names[class_num])
    plt.show()


#    2)Box Plot
def boxPlot(data):
    data = data[classes[class_num][0]:classes[class_num][1]]            #cuts out the data from the classes you don't want to graph
    plt.title(names[index_1] + ' For Class ' + str(class_num+1))
    plt.ylabel('y-axis')
    plt.boxplot(data)
    plt.show()

#####################################################################################  Question 2

#    1)Correlation Plots
def correlation(x,y):
    x_mean = np.mean(x)
    y_mean = np.mean(y)
    x_std = 0
    y_std = 0
    cov = 0
    for z in range(0,len(x)):                           #summing the deviation and covariance 
        x_std += np.square(x[z] - x_mean)
        y_std += np.square(y[z] - y_mean)
        cov += (x[z] - x_mean)*(y[z] - y_mean)
    cov = cov / (len(x)-1)                              #averaging
    x_std = x_std / (len(x)-1)                          #averaging
    y_std = y_std / (len(x)-1)                          #averaging
    x_std = np.sqrt(x_std)
    y_std = np.sqrt(y_std)
    #x_std = np.std(x)      # these didn't give the right answer
    #y_std = np.std(y)
    p = cov/(x_std * y_std)                             #calculating the correlation
    return p
    
def corMatrix(data):
    corMat = np.zeros((length,length))
    temp = 0
    for y in range(length):
        for x in range(y,length):
            temp = correlation(data[y],data[x])
            corMat[y][x] = temp                     #saving computing time
            corMat[x][y] = temp
    names_c = names
    if i:
        names_c.pop(4)      #remove class from iris dataset
    else:                   #remove class from wine dataset
        names_c.pop(0)
        corMat = np.delete(corMat,0,1)          #in the main function the iris names never get passed into data, unlike the class for the wine dataset
        corMat = np.delete(corMat,0,0)          #so for the wine you need to delete the row and column
    plt.title('Correlation Matrix')
    plt.imshow(corMat, cmap='RdBu', interpolation='nearest')
    plt.yticks(np.arange(len(names_c)),names_c)
    plt.xticks(np.arange(len(names_c)),names_c, rotation='vertical')
    plt.colorbar()
    plt.show()

#    2)Scatter Plot
def scatterPlot(a,b):
    one = plt.scatter(a[0:49],b[0:49],c="r")                    #plotting the dots and assigning color
    two = plt.scatter(a[50:99],b[50:99],c="b")                  #plotting the dots and assigning color
    three = plt.scatter(a[100:149],b[100:149],c="g")            #plotting the dots and assigning color
    title = names[index_1] + ' vs ' + names[index_2]
    plt.xlabel(names[index_1])
    plt.ylabel(names[index_2])
    plt.title(title)
    plt.legend((one,two,three),('Iris-setosa','Iris-versicolor','Iris-virginica'),
    scatterpoints=1,loc='lower right',ncol=1,fontsize=8)        #creating the legend
    plt.grid(True)
    plt.show()


#    3)Distance
def distance(x,y,p):
    total = 0
    for i in range(length):                 #summing
        total += abs(x[i] - y[i])**p
    total = total**(1/float(p))             #root of p
    return total
    
def dpMatrix(data,h):
    dpMat = np.zeros((h,h))
    dpMat2= np.zeros((h,h))
    #ret = []
    rope2 = ''
    #I wasn't sure if we were suppose to do part e only in the report or our code should do it.
    #So my code calculates both data point matrix with p order = 1 and 2. But only shows the one you asked for
    #so that's why they're are two dpMats
    temp = 0                                        #reduces the number of calls to the distance() function
    for x in range(h):                              #fills out dpMats[][]
        for y in range(x,h):
            temp = distance(data[:,x],data[:,y],1)
            dpMat[x][y] = temp
            dpMat[y][x] = temp
            temp = distance(data[:,x],data[:,y],2)
            dpMat2[x][y] = temp
            dpMat2[y][x] = temp
    ntndp = np.zeros((h,6))                         #this was mostly for debugging purposes, (stands for Non Trivial Nearest Data Point)
    #0.index    
    #1.nearest data point with p = 1     
    #2. with p = 2       
    #3. 0 = different nearest data points when p changes     1 = same nears dp 
    #4. when p = 1: 1 = the two datapoints are the same class     0 = diff
    #5. when p = 2: 1 = the two datapoints are the same class     0 = diff
    for x in range(h):
        ntndp[x][0] = x
        temp = 999999
        temp2 = 999999
        index_t = 0
        index_t2 = 0
        for y in range(h):                      #nested for loop, iterates to find the closest
            if x != y:
                if temp > dpMat[x][y]:          #evaluates whether if dpMat[x][y] is a closer datapoint
                    temp = dpMat[x][y]
                    index_t = y
                if temp2 > dpMat2[x][y]:
                    temp2 = dpMat2[x][y]
                    index_t2 = y
        ntndp[x][1] = index_t                   #nearest data point with p = 1   
        ntndp[x][2] = index_t2                  #nearest data point with p = 2
        if index_t == index_t2:
            ntndp[x][3] = 1                     #nearest data point is the same regardless of whether p = 1 or 2
    
    percent_class = 0
    percent_p = 0
    for x in range(h):                          #for printing out the nearest data point list
        y2 = 0          #index of the class name of the data point
        z = 0           #index of the class name of the nearest data point
        for y in range(len(classes)):           #for determining the class name
            
            if classes[y][0] <= ntndp[x][0] and ntndp[x][0] <= classes[y][1]:
                if classes[y][0] <= ntndp[x][1] and ntndp[x][1] <= classes[y][1]:
                    ntndp[x][4] = 1             #print('same class with p = 1')
                if classes[y][0] <= ntndp[x][2] and ntndp[x][2] <= classes[y][1]:
                    ntndp[x][5] = 1             #print('same class with p = 2')
                y2 = y                          #index set
            if classes[y][0] <= ntndp[x][1] and ntndp[x][1] <= classes[y][1] and p_order == 1: #p = 1
                z = y                           #index set
            if classes[y][0] <= ntndp[x][2] and ntndp[x][2] <= classes[y][1] and p_order == 2: #p = 2
                z = y                           #index set
        
        #originally my code would print the variable rope out to show each points 
        
        rope = str(int(ntndp[x][0])) + '[' + class_names[y2] +'] -> '               #output
        rope = rope + str(int(ntndp[x][p_order])) + '[' + str(class_names[z])+ ']'  #output
        rope2 = rope2 + rope + '\n'
        
        if ntndp[x][p_order + 3] == 0:                              #the 2 data points are from different class
            rope = rope + ' DIFF '
            percent_class += 1
        else:
            rope = rope + ' SAME '                                  #the 2 data points are from the same class
        if ntndp[x][3] == 0:
            rope = rope + ' P CHANGES '                             #changing p affects the nearest data point
            percent_p += 1
        else:
            rope = rope + ' NO CHANGES'                             #changing p doesn't affect the nearest data point
        
        
    percent_class =  100 * float(percent_class) / float(h)
    percent_p = 100 * float(percent_p) / float(h)
    print('Percentage of nearest data points being from different classes ' + str(percent_class) + '%')
    print('Percentage of nearest data points changing when p changes ' + str(percent_p) + '%')
    if p_order == 1:                                                #only plots the one the user wanted
        plt.imshow(dpMat, cmap='RdBu', interpolation='nearest')
    else:
        plt.imshow(dpMat2, cmap='RdBu', interpolation='nearest')
    plt.title('Data Point x Data Point matrix with p = ' + str(p_order))
    plt.colorbar()
    plt.show()
    return rope2


###################################### Main
if i:
    f = open("python\iris.txt","r")
else:
    f = open("python\wine.txt","r")
f1 = f.readlines()
h = len(f1)
data = np.zeros((length,h))
x_index = 0
temp = 0
for x in f1:
    line = x.split(',')
    if i:
        temp = len(line)-1          #remove the class name from the iris data set, since data is type float and you can't cast a str to float
    else:
        temp = len(line)            #this is to keep Proline in the wine data set
    
    for y in range(temp):
        data[y][x_index] = line[y]  #fillout the matrix
    x_index+= 1

#calls the appropriate function
if prob == '1a':
    histogram(data[index_1])
elif prob == '1b':
    boxPlot(data[index_1])
elif prob == '2a':
    corMatrix(data)
elif prob == '2b':
    scatterPlot(data[index_1],data[index_2])
elif prob == '2c':
    ans = raw_input('Would you like to print the list of nearest neighbors(y/n) ')
    ret = dpMatrix(data,h)
    if ans == 'y':
        print('\nPrinting nearest pairs: ')
        print(ret)
    
