import os
import sys
# os.append('./')
import numpy as np
from utils import IoU
import cv2

file = 'test_kitti_train.txt'
target_neg_dir = "kitti_dir/neg"
target_pos_dir = "kitti_dir/pos"
with open(file) as f:
	labels = f.readlines()
print("num ", len(labels))
pos_id = 0
neg_id = 0

for idx, label in enumerate(labels):
	print("{}/{}".format(idx+1, len(labels)))
	img_path = label.split(' ')[0]
	print(img_path)
	# bbox = [int(float(ele)) for ele in label.strip().split(' ')[1:]]
	bbox = [ele for ele in label.strip().split(' ')[1:]]
	# print(bbox)
	bbox = np.array(bbox, dtype = np.float32).reshape(-1,4)
	img = cv2.imread(img_path)
	h, w, c = img.shape
	img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
	# for box in bbox:
	#     cv2.rectangle(img_color, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), (0, 0, 255), 1)

	# cv2.imwrite(os.path.join('kitti_dir', str(uuid.uuid4()) + ".jpg"), img_color)

	# while neg_num < 200:
	#     neg_num += 1
	#     if neg_num % 10 == 0:
	#         print(neg_num)
	#     size_w = np.random.randint(w-5, w+5)
	#     size_h = np.random.randint(h-10, h+10)
	#     if size_h < size_w:
	#         continue
	#     x_c = np.random.randint(0, w-2)
	#     y_c = np.random.randint(0, h-2)
	#     crop_box = np.array([x_c, y_c, (x_c + size_w), (y_c + size_h)])
	#     Iou = IoU(crop_box,bbox)
	#     # print(Iou)

	#     corpped_img = img[y_c : (y_c + size_h), x_c : (x_c + size_w), : ]
	#     # cv2.imwrite('./1.jpg',corpped_img)
	#     # resized_crop_img = cv2.resize(corpped_img,(12,12), interpolation=cv2.INTER_LINEAR)

	#     if np.max(Iou) >= 0.65:
	#         # pos_txt.write(os.path.join(pos_dir, str(p_idx) + '.jpg') + ' 1 %0.3f %0.3f %0.3f %0.3f\n'%(offset_x1, offset_y1, offset_x2, offset_y2))
	#         cv2.imwrite(os.path.join(target_pos_dir, str(pos_id) + '.jpg'), corpped_img)
	#         # ori_file_12.write(img_path + " " + os.path.join(pos_dir,str(p_idx) + ".jpg 0") + '\n')
	#         pos_id += 1
	#     elif np.max(Iou) <= 0.3:
	#         # par_txt.write(os.path.join(par_dir, str(d_idx) + ".jpg") + ' -1 %0.3f %0.3f %0.3f %0.3f\n'%(offset_x1, offset_y1, offset_x2, offset_y2))
	#         cv2.imwrite(os.path.join(target_neg_dir, str(neg_id) + ".jpg"), corpped_img)
	#         # ori_file_12.write(img_path + " " + os.path.join(par_dir,str(d_idx) + ".jpg 0") + '\n')
	#         neg_id += 1
		  
	#     print("images done, positive: %s negative: %s total: %s"%(pos_id,neg_id, (pos_id+neg_id)))
		

	x1, y1, x2, y2 = bbox[0]

	neg_num = 0
	while neg_num < 40:
		neg_num += 1

		x_c = np.random.randint(0, int(w*0.8))
		y_c = np.random.randint(0, int(h*0.8))
		size_w = np.random.randint(x2-x1-30, x2-x1+30)
		size_h = np.random.randint(y2-y1-30, y2-y1+30)
		if size_h <= size_w*0.7 or size_w <= 0 or size_h <= 0 or (x_c + size_w) >= w or (y_c + size_h) >= h:
			continue
		
		
		crop_box = np.array([x_c, y_c, (x_c + size_w), (y_c + size_h)])

		Iou = IoU(crop_box,bbox)
		if np.max(Iou) < 0.3:
			corpped_img = img[y_c:(y_c + size_h), x_c:(x_c + size_w),:]
			# if neg_id == 20000:
			# 	continue
			# par_txt.write(os.path.join(par_dir, str(d_idx) + ".jpg") + ' -1 %0.3f %0.3f %0.3f %0.3f\n'%(offset_x1, offset_y1, offset_x2, offset_y2))
			cv2.imwrite(os.path.join(target_neg_dir, str(neg_id) + ".jpg"), corpped_img)
				# ori_file_12.write(img_path + " " + os.path.join(par_dir,str(d_idx) + ".jpg 0") + '\n')
			neg_id += 1










	for box in bbox:
		#box (xmin, ymin, xmax, ymax)
		x1, y1, x2, y2 = box
		w_ = x2 - x1 + 1
		h_ = y2 - y1 + 1
		##if img too small ignore it
		if x2 < x1 or y2 < y1:
			continue
		if max(w_, h_) < 10 or x1 <0 or y1 <0:
			continue
	##generate positive.txt
		for i in range(17):
			# print(i)
			size_w = np.random.randint(w_ - 10, w_ + 10)
			size_h = np.random.randint(h_ - 10, h_ + 10) 
	  
			delta_x = np.random.randint(x1 - 10, x1 + 10)
			delta_y = np.random.randint(y1 - 10, y1 + 10)

			nx1 = max(delta_x, 0)
			ny1 = max(delta_y, 0)
			nx2 = min(nx1 + size_w, w)
			ny2 = min(ny1 + size_h, h)

			if nx2 >= w or ny2 >= h or nx2 <= nx1 or ny2 <= ny1:
				continue
			crop_box = np.array([nx1, ny1, nx2, ny2])

		   
			corpped_img = img[int(ny1) : int(ny2), int(nx1) : int(nx2), :]
	
			# resized_crop_img = cv2.resize(corpped_img, (12, 12),interpolation=cv2.INTER_LINEAR)
			box_ = box.reshape(1,-1)

			if IoU(crop_box, box_) >= 0.65:
				# if pos_id == 20000:
				#     continue
				# pos_txt.write(os.path.join(pos_dir, str(p_idx) + '.jpg') + ' 1 %0.3f %0.3f %0.3f %0.3f\n'%(offset_x1, offset_y1, offset_x2, offset_y2))
				cv2.imwrite(os.path.join(target_pos_dir, str(pos_id) + '.jpg'), corpped_img)
				# ori_file_12.write(img_path + " " + os.path.join(pos_dir,str(p_idx) + ".jpg 0") + '\n')
				pos_id += 1

			elif IoU(crop_box, box_) <=0.2:
				# if neg_id == 20000:
				#     continue
				continue
				# par_txt.write(os.path.join(par_dir, str(d_idx) + ".jpg") + ' -1 %0.3f %0.3f %0.3f %0.3f\n'%(offset_x1, offset_y1, offset_x2, offset_y2))
				# cv2.imwrite(os.path.join(target_neg_dir, str(neg_id) + ".jpg"), corpped_img)
				# ori_file_12.write(img_path + " " + os.path.join(par_dir,str(d_idx) + ".jpg 0") + '\n')
				# neg_id += 1
			# if neg_id ==  pos_id:
			#     quit()
			print("images done, positive: %s negative: %s total: %s"%(pos_id,neg_id, (pos_id+neg_id)))

