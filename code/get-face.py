#! /bin/env/ python3

import os
import cv2
import pickle
from collections import defaultdict
from progressbar import progressbar

def supply_dict(key = 'genders',
                 legend = {'genders' : 0, 'emotions' : 1}):
    if legend[key] == 0:
        return dict([
            ('M', supply_dict('emotions')),
            ('F', supply_dict('emotions')),
        ])
    else:
        return dict([
            ('N', defaultdict(str)),
            ('F', defaultdict(str)),
            ('A', defaultdict(str)),
            ('HO', defaultdict(str)),
            ('HC', defaultdict(str)),
        ])

class face_provider:

    data_path = '../data/cfd2.0.3/images'

    def __init__(self, path=data_path):
        self.image_containers = os.listdir(os.path.abspath(path))
        self.images = {
            'A' : supply_dict(), 'W' : supply_dict(),
            'B' : supply_dict(), 'L' : supply_dict(),
        }
        self.indexed_faces = set()

    def load_from_pickle(self, path="pickled/"):
        with open(os.path.join(path, "images.pickle"), 'rb') as file:
            self.images = pickle.load(file)
        with open(os.path.join(path, "image_containers.pickle"), 'rb') as file:
            self.image_containers = pickle.load(file)
        with open(os.path.join(path, "indexed_faces.pickle"), 'rb') as file:
            self.indexed_faces = pickle.load(file)

    def dump_to_pickle(self, path="pickled/"):
        with open(os.path.join(path, "images.pickle"), 'wb') as file:
            pickle.dump(self.images, file)
        with open(os.path.join(path, "image_containers.pickle"), 'wb') as file:
            pickle.dump(self.image_containers, file)
        with open(os.path.join(path, "indexed_faces.pickle"), 'wb') as file:
            pickle.dump(self.indexed_faces, file)

    def index_faces(self,
                     path = data_path):
                     # races = {'A', 'W', 'B', 'L'},
                     # genders = {'M', 'F'},
                     # emotions = {'A', 'F', 'N', 'HO', 'HC'}):

        print("Indexing faces")

        for container_name in progressbar(self.image_containers, redirect_stdout=True):
            if len(container_name)<1 or len(container_name.split('-'))<2:
                print("ignoring", container_name)
                continue
            rac, gen = container_name[:2]
            for filename in os.listdir(os.path.join(os.path.abspath(path), container_name)):
                basename = filename.split('.')[0]
                if not len(basename):
                    continue
                id,emo = basename.split('-')[2], basename.split('-')[4]
                # print(os.path.join(os.path.abspath(path), filename))
                self.images[rac][gen][emo][id] = cv2.imread(os.path.join(os.path.abspath(path), container_name, filename), 0)
                self.crop_square(rac, gen, emo, id)
                self.resize(rac, gen, emo, id, 100)
                self.indexed_faces.add(rac+gen+emo+id)
                cv2.imwrite("../data/processed_dump/%s"%filename, self.images[rac][gen][emo][id])

    def crop_square(self, rac='W', gen='F', emo='HC', id='022'):
        img = self.images[rac][gen][emo][id]
        h,w = img.shape[0:2]
        d = 0.5 * abs(h-w)
        # print(h,w,d)
        cropped_img = img[int((h>w)*d):int(h-(h>w)*d),
                          int((w>h)*d):int(w-(w>h)*d)]
        self.images[rac][gen][emo][id] = cropped_img
        # return cropped_img

    def resize(self, rac='W', gen='F', emo='HC', id='022', dim=100):
        img = self.images[rac][gen][emo][id]
        img = cv2.resize(img, (dim,dim), interpolation = cv2.INTER_AREA)
        self.images[rac][gen][emo][id] = img

    def get_face(self, rac='W', gen='F', emo='HC', id='022'):
        return self.images[rac][gen][emo][id]

    def list_faces(self, rac='W', gen='F', emo='HC'):
        return self.images[rac][gen][emo]

    # def make_grayscale(self, rac='W', id='022', gen='F', emo='HC'):
    #     img = self.images[rac][id][gen][emo]
    #     img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

fp = face_provider()
fp.index_faces()
fp.dump_to_pickle()
# with open('face_provider.get-face.py.pickle', 'wb') as file:
#     pickle.dump(fp, file)
# print(fp.list_faces('W','M','HC'))
