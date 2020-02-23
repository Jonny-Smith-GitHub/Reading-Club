# module import
import random
import math
import numpy as np
from matplotlib import pyplot as plt

# parameters initilization
data=[]
SSE_list=[]
epoch=50
dimension=10
size=100
result_list=[]
k_range=10

def point_generator(n):
    # generate 1000 random points of n dimension
    for _ in range(0,size):
        random_point=[]
        for _ in range(0,n):
            random_point.append(random.random())
        data.append(random_point)

# generate random points
point_generator(dimension)

def k_means(k):
    # k-means
    centroid_list=random.sample(data,k)
    # pick random samples as original centroids

    for _ in range(0,epoch):
        # for every echo
        SSE_list=[0]*k
        next_centroid_list=[[0]*dimension]*k
        count_list=[0]*k

        for i in range(0,size):
            # for every sample
            distance_list=[0]*k
            
            for j in range(0,k):
                # for every centroid
                distance_list[j]=pow(np.linalg.norm(np.array(data[i])-np.array(centroid_list[j])),2)
                # compute the distance from sample

            min_index=distance_list.index(min(distance_list))
            minmum=min(distance_list)
            # determine the centroid this sample belong

            SSE_list[min_index]+=minmum
            count_list[min_index]+=1
            next_centroid_list[min_index]=np.array(data[i])+np.array(next_centroid_list[min_index])

        for i in range(0,k):
            centroid_list[i]=np.array(next_centroid_list[i])/count_list[i]
            # determine the next centroids

    result_list.append(sum(SSE_list)) 
    # pick random samples as original centroids

for k in range(1,k_range):
    k_means(k)

#visualization
plt.xlabel("cluster number")
plt.ylabel("SSE")
x=range(1,k_range)
y=result_list
plt.plot(x,y)
plt.plot(x,y,'oc')
plt.show()

# determine the best k using elbow rule
cos_list=[]

for i in range(1,k_range-2):
    cos=(result_list[i]*result_list[i]+result_list[i-1]*result_list[i+1]\
        -result_list[i-1]*result_list[i]-result_list[i]*result_list[i+1])\
        /2*pow((1+pow((result_list[i-1]-result_list[i]),2)),1/2)*\
        pow((1+pow((result_list[i+1]-result_list[i]),2)),1/2)
    cos_list.append(cos)

index=cos_list.index(max(cos_list))+1

print('The best cluster number is',index)