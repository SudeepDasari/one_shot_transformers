import cv2
import glob
import xml.etree.ElementTree as ET 
import os
from multiprocessing import Pool
import numpy as np
import imageio


def parse_xml(path):
    tree = ET.parse(open(path, 'r'))
    root = tree.getroot()
    return list(root)


def numeric_sort(files):
    return sorted(files, key=lambda x: int(x.split('/')[-1].split('.')[0]))


base_folder = '/home/sudeep/hdd/EPIC'

# for i in range(1, 32):
#     P = 'P{:02d}'.format(i)
#     if not (os.path.exists('{}/MiniImages/{}'.format(base_folder, P)) and os.path.exists('{}/MiniAnnotations/{}'.format(base_folder, P))):
#         continue

#     sub_folder = glob.glob('{0}/MiniImages/{1}/*'.format(base_folder, P))[0].split('/')[-1]
#     annotations = [parse_xml(f) for f in numeric_sort(glob.glob('{0}/MiniAnnotations/{1}/{2}/*.xml'.format(base_folder, P, sub_folder)))]

#     color_mappings = {}

#     for objects, file_path in zip(annotations, numeric_sort(glob.glob('{0}/MiniImages/{1}/{2}/*.jpg'.format(base_folder, P, sub_folder)))):
#         img = cv2.imread(file_path)
#         cv2.putText(img,file_path,(20,20),0,0.7, (0,0,0))

#         for obj in objects:
#             name_id = [o for o in obj if o.tag == 'name_id'][0].text
#             name = [o for o in obj if o.tag == 'name'][0].text

#             bound_box = [o for o in obj if o.tag == 'bndbox'][0]
#             bound_box = {b.tag: int(float(b.text)) for b in bound_box}
#             if name_id in color_mappings:
#                 color = color_mappings[name_id]
#             else:
#                 color = (np.random.randint(200), np.random.randint(200), np.random.randint(200))
#                 color_mappings[name_id] = color

#             cv2.rectangle(img,(bound_box['xmin'], bound_box['ymin']),(bound_box['xmax'] , bound_box['ymax']),color,2)
#             cv2.putText(img, name,(bound_box['xmin'] + 20, bound_box['ymin'] + 20),0, 0.7, color)

#         cv2.imshow('viewer', img)
#         cv2.waitKey(0)


def load_video(folder, T=100, target_shape=(224, 224)):
    im_files = numeric_sort(glob.glob(folder + '/*.jpg'))
    if len(im_files) < T:
        print('{} only has {} frames'.format(folder, len(im_files)))
        replace=True
    else:
        replace=False
    chosen_ims = np.sort(np.random.choice(len(im_files), size=T, replace=replace))
    ims = [cv2.imread(im_files[f_i])[:,:,::-1] for f_i in chosen_ims]
    h, w = ims[0].shape[:2]
    w_start = np.random.randint(w - h)

    for i, f in enumerate(ims):
        ims[i] = cv2.resize(f[:, w_start:w_start+h], target_shape, interpolation=cv2.INTER_AREA)
    return ims


N = 200
all_videos = glob.glob(base_folder + '/MiniImages/*/*')
v_f = []
for f in all_videos:
    vid_len = len(glob.glob(f + '/*.jpg'))
    v_f.append((f, vid_len))
    print('{} has {} frames'.format(f, vid_len))

for n in [50, 100, 200, 300, 400, 500]:
    filted = [v[1] for v in v_f if v[1] >= n]
    print('{} have at least {} frames totalling {} frames'.format(len(filted), n, sum(filted)))

T = 200
all_videos = [f for f, l in v_f if l >= T]
for n in range(N):
    f_n, s_n = np.random.choice(len(all_videos), size=2, replace=False)
    first, second = [load_video(all_videos[f], T) for f in (f_n, s_n)]
    
    f_n, s_n = [all_videos[f].split('/')[-1] for f in (f_n, s_n)]
    writer = imageio.get_writer('{}_{}.gif'.format(f_n, s_n))
    for f, s in zip(first, second):
        writer.append_data(np.concatenate((f, s), 1))
    writer.close()
