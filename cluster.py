# -*- coding: utf-8 -*-
import argparse
import numpy as np
# from itertools import izip
import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import glob

from cfgs.config import cfg

# Original code @ferada http://codereview.stackexchange.com/questions/128315/k-means-clustering-algorithm-implementation

best_clusters = []
best_avg_iou = 0
best_avg_iou_iteration = 0

def overlap(x1, w1, x2, w2):
    l1 = x1 - w1/2.
    l2 = x2 - w2/2.
    left = l1 if l1 > l2 else l2
    r1 = x1 + w1/2.
    r2 = x2 + w2/2.
    right = r1 if r1 < r2 else r2
    return right - left

def intersection(a, b):
    w = overlap(a[0], a[2], b[0], b[2])
    h = overlap(a[1], a[3], b[1], b[3])
    if w < 0 or h < 0:
        return 0
    return w*h

def area(x):
    return x[2]*x[3]

def union(a, b):
    return area(a) + area(b) - intersection(a, b)

def iou(a, b):
    return intersection(a, b) / union(a, b)

def niou(a, b):
    return 1. - iou(a,b)

def equals(points1, points2):
    if len(points1) != len(points2):
        return False

    for p_idx, point1 in enumerate(points1):
        point2 = points2[p_idx]
        if point1[0] != point2[0] or point1[1] != point2[1] or point1[2] != point2[2] or point1[3] != point2[3]:
            return False

    return True

def compute_centroids(clusters):
    return [np.mean(cluster, axis=0) for cluster in clusters]

def closest_centroid(point, centroids):
    min_distance = float('inf')
    belongs_to_cluster = None
    for j, centroid in enumerate(centroids):
        dist = niou(point, centroid)

        if dist < min_distance:
            min_distance = dist
            belongs_to_cluster = j

    return belongs_to_cluster, min_distance

def kmeans(k, centroids, points, iter_count=0, iteration_cutoff=25):
    global best_clusters
    global best_avg_iou
    global best_avg_iou_iteration
    clusters = [[] for _ in range(k)]
    clusters_iou = []
    clusters_niou = []

    for point in points:
        idx, dist = closest_centroid(point, centroids)
        clusters[idx].append(point)
        clusters_niou.append(dist)
        clusters_iou.append(1.-dist)

    avg_iou = np.mean(clusters_iou)
    if avg_iou > best_avg_iou:
        best_avg_iou = avg_iou
        best_clusters = clusters
        best_avg_iou_iteration = iter_count

    print("Iteration {}".format(iter_count))
    print("Average iou to closest centroid = {}".format(avg_iou))
    print("Sum of all distances (cost) = {}\n".format(np.sum(clusters_niou)))

    new_centroids = compute_centroids(clusters)

    for i in range(len(new_centroids)):
        shift = niou(centroids[i], new_centroids[i])
        print("Cluster {} size: {}".format(i, len(clusters[i])))
        print("Centroid {} distance shift: {}\n\n".format(i, shift))

    return (new_centroids, iter_count, best_avg_iou_iteration)

def plot_anchors(pascal_anchors):
    fig1 = plt.figure()
    ax1 = fig1.add_subplot(111, aspect='equal')
    ax1.set_ylim([0,500])
    ax1.set_xlim([0,900])

    for i in range(len(pascal_anchors)):
        bbox = pascal_anchors[i]

        lower_right_x = bbox[0]-(bbox[2]/2.0)
        lower_right_y = bbox[1]-(bbox[3]/2.0)

        ax1.add_patch(
            patches.Rectangle(
                (lower_right_x, lower_right_y),
                bbox[2],
                bbox[3],
                facecolor="blue"
            )
        )
    plt.show()

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--data_files', help='comma separated data file list', required=True)
    args = parser.parse_args()


    data_files = args.data_files.split(',')
    # data_files.append("voc_2007_train.txt")
    # data_files.append("voc_2007_val.txt")
    # data_files.append("voc_2007_test.txt")
    # data_files.append("voc_2012_train.txt")
    # data_files.append("voc_2012_val.txt")
    # data_files.append("cmdt_train.txt")
    # data_files.append("draw_train.txt")

    lines = []
    for data_file in data_files:
        with open(data_file) as f:
            lines.extend(f.readlines())

    print("Number of images: " + str(len(lines)))

    box_data = []
    for idx, line in enumerate(lines):

        if idx % 1000 == 0 and idx > 0:
            print(str(idx))

        record = line.split(' ')
        record[1:] = [float(num) for num in record[1:]]


        image = cv2.imread(record[0])
        # print(record[0])
        s = image.shape
        h = float(s[0])
        w = float(s[1])

        width_rate = int(cfg.img_w / cfg.grid_w) / w
        height_rate = int(cfg.img_h / cfg.grid_h) / h

        i = 1
        while i < len(record):
            xmin = record[i]
            ymin = record[i + 1]
            xmax = record[i + 2]
            ymax = record[i + 3]
            i += 5

            x_center = (xmin + xmax) / 2
            y_center = (ymin + ymax) / 2
            width = (xmax - xmin + 1) * width_rate
            height = (ymax - ymin + 1) * height_rate
            box_data.append([x_center, y_center, width, height])

    box_data = np.array(box_data)

    for i in range(len(box_data)):
        box_data[i][0] = 0
        box_data[i][1] = 0

    # k-means picking the first k points as centroids
    k = 5
    centroids = box_data[:k]
    iter_count = 0
    iteration_cutoff = 5
    while True:
        (centroids, iter_count, best_avg_iou_iteration) = kmeans(k, centroids, box_data, iter_count, iteration_cutoff)
        if iter_count >= best_avg_iou_iteration + iteration_cutoff:
            break
        iter_count += 1

    # Get anchor boxes from best clusters
    box_anchors = np.asarray([np.mean(cluster, axis=0) for cluster in best_clusters])

    # Sort by width
    box_anchors = box_anchors[box_anchors[:, 2].argsort()]

    print("k-means clustering pascal anchor points (original coordinates)")
    print("Found at iteration {} with best average IoU: {} ".format(best_avg_iou_iteration, best_avg_iou))
    print("{}".format(box_anchors[:,2:]))

    # Hardcode anchor center coordinates for plot
    box_anchors[3][0] = 250
    box_anchors[3][1] = 100
    box_anchors[0][0] = 300
    box_anchors[0][1] = 450
    box_anchors[4][0] = 650
    box_anchors[4][1] = 250
    box_anchors[2][0] = 110
    box_anchors[2][1] = 350
    box_anchors[1][0] = 300
    box_anchors[1][1] = 300

    box_anchors[:, 2] = box_anchors[:, 2] / width_rate
    box_anchors[:, 2] = box_anchors[:, 3] / height_rate
    plot_anchors(box_anchors)
