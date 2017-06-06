'''
Created on Feb 20, 2017

@author: jumabek
'''
from os import listdir
from os.path import isfile, join
import argparse
import cv2
import numpy as np
import sys
import os
import shutil
import random
      

def IOU(x,centroids):
    dists = []
    for centroid in centroids:
        c_w,c_h = centroid
        w,h = x
        if c_w>=w and c_h>=h:
            dist = w*h/(c_w*c_h)
        elif c_w>=w and c_h<=h:
            dist = w*c_h/(w*h + (c_w-w)*c_h)
        elif c_w<=w and c_h>=h:
            dist = c_w*h/(w*h + c_w*(c_h-h))
        else: #means both w,h are bigger than c_w and c_h respectively
            dist = (c_w*c_h)/(w*h)
        dists.append(dist)
    return np.array(dists)

def avg_IOU(X,centroids):
    n,d = X.shape
    sum = 0.
    for i in range(X.shape[0]):
        #note IOU() will return array which contains IoU for each centroid and X[i] // slightly ineffective, but I am too lazy
        sum+= max(IOU(X[i],centroids)) 
    return sum/n

def write_anchors_to_file(centroids,X,anchor_file):
    f = open(anchor_file,'w')
    
    anchors = centroids*416/32
    
    print('Anchors = ', centroids*416/32)
    
    num_anchors = anchors.shape[0]
    for i in range(num_anchors-1):
        f.write('%f,%f, '%(anchors[i][0],anchors[i][1]))

    #there should not be comma after last anchor, that's why
    f.write('%f,%f\n'%(anchors[num_anchors-1][0],anchors[num_anchors-1][1]))
    
    f.write('%f\n'%(avg_IOU(X,centroids)))

def kmeans(X,centroids,eps,anchor_file):
    
    D=[]
    old_D = []
    iterations = 0
    diff = 1e5
    c,dim = centroids.shape

    while True:
        iterations+=1
        D = []            
        for i in range(X.shape[0]):
            d = 1 - IOU(X[i],centroids)
            D.append(d)
        D = np.array(D)
        if len(old_D)>0:
            diff = np.sum(np.abs(D-old_D))
        
        print('diff = %f'%diff)

        if diff<eps or iterations>100:
            print("Number of iterations took = %d"%(iterations))
            print("Centroids = ",centroids)

            
            write_anchors_to_file(centroids,X,anchor_file)
            
            return

        #assign samples to centroids 
        belonging_centroids = np.argmin(D,axis=1)
        print(belonging_centroids )

        #calculate the new centroids
        centroid_sums=np.zeros((c,dim),np.float)
        for i in range(belonging_centroids.shape[0]):
            centroid_sums[belonging_centroids[i]]+=X[i]
        
        for j in range(c):
            
            print('#annotations in centroid[%d] is %d'%(j,np.sum(belonging_centroids==j)))
            centroids[j] = centroid_sums[j]/np.sum(belonging_centroids==j)
        
        print('new centroids = ',centroids        )



        old_D = D.copy()
    print(D)

def main():

    pascal_data = []
    
    grid_w = 13
    grid_h = 13

    data_files = []
    data_files.append("voc_2007_train.txt")
    data_files.append("voc_2007_val.txt")
    data_files.append("voc_2007_test.txt")
    data_files.append("voc_2012_train.txt")
    data_files.append("voc_2012_val.txt")

    lines = []
    for data_file in data_files:
        with open(data_file) as f:
            lines.extend(f.readlines())

    print("Number of images: " + str(len(lines)))

    for idx, line in enumerate(lines):

        if idx % 100 == 0 and idx > 0:
            print(str(idx))

        record = line.split(' ')
        record[1:] = [float(num) for num in record[1:]]

        image = cv2.imread(record[0])
        s = image.shape
        h = float(s[0])
        w = float(s[1])

        width_rate = grid_w / w
        height_rate = grid_h / h

        i = 1
        while i < len(record):
            xmin = record[i]
            ymin = record[i + 1]
            xmax = record[i + 2]
            ymax = record[i + 3]
            i += 5

            x_center = (xmin + xmax) / 2
            y_center = (ymin + ymax) / 2
            width = xmax - xmin
            height = ymax - ymin
            width = (xmax - xmin) * width_rate
            height = (ymax - ymin) * height_rate
            pascal_data.append([width, height])

    annotation_dims = np.array(pascal_data)

    eps = 0.005
    
    # anchor_file = join( args.output_dir,'anchors%d.txt'%(args.num_clusters))
    indices = [ random.randrange(annotation_dims.shape[0]) for i in range(5)]
    centroids = annotation_dims[indices]
    kmeans(annotation_dims,centroids,eps,"output.txt")

    print('centroids.shape', centroids.shape)

if __name__=="__main__":
    main()