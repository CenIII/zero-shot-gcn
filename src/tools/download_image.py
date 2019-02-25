import argparse
import os
import threading
# import urllib
import glob

import shutil
import tempfile
import urllib.request
import pickle
import cv2
import numpy as np
import tqdm

import io
from PIL import Image
import warnings
from multiprocessing import Process

import signal
import sys
def signal_handler(sig, frame):
        print('You pressed Ctrl+C!')
        for p in procList:
            p.terminate()
        sys.exit(0)
signal.signal(signal.SIGINT, signal_handler)

data_dir = '../data/'
warnings.filterwarnings("ignore", "Possibly corrupt EXIF data", UserWarning)
warnings.filterwarnings("ignore", "Corrupt EXIF data", UserWarning)
procList = []


def download(vid_file):
    with open(vid_file) as fp:
        vid_list = [line.strip() for line in fp]

    # url_list = 'http://www.image-net.org/download/synset?wnid='
    # url_key = '&username=%s&accesskey=%s&release=latest&src=stanford' % (args.user, args.key)

    # testfile = urllib.URLopener()

    # todo: read url lists
    # fall11_file = data_dir+'list/fall11_urls.txt'
    # url_list = {} # {n00145923:['http://...','...']}
    # with open(fall11_file,'r',errors='replace') as f:
    #     lines = f.readlines()
    #     print("done reading.")
    #     print("num of lines: "+str(len(lines)))
    #     qdar = tqdm.tqdm(lines,total=len(lines),ascii=True)
    #     for line in qdar:
    #         # print(line)
    #         line_sp = line.split('\t')
    #         if len(line_sp)>2:
    #             continue
    #         name, url = line_sp
    #         class_name, image_name = name.split('_')
    #         ent = url_list.setdefault(class_name,[])
    #         ent.append(url[:-1])
    # with open(data_dir+'list/url_list','wb') as f:
    #     pickle.dump(url_list,f)

    # print("done.")
    with open(data_dir+'list/url_list','rb') as f:
        url_list = pickle.load(f)

    def get_wnid_url_addr(wnid):
        # todo: return matching url list for the wnid
        if wnid in url_list:
            return url_list[wnid]
        else:
            print("no such class.")
            return None
        
    
    def download_pack_save(wnid_url_list,save_dir,index,valid_ind,pid,qdar):
        image_list = []
        outer_ind = 0
        inner_ind = 0
        length = len(wnid_url_list)
        os.makedirs(save_dir+'.proc',exist_ok=True)
        for url in wnid_url_list:
            index += 1
            outer_ind += 1
            try:
                img = urllib.request.urlopen(url, timeout=4).read()
                if img[:4] == b'\xff\xd8\xff\xe0': # is image
                    valid_ind += 1
                    inner_ind += 1
                    target = os.path.join(save_dir,str(inner_ind)+'.JPG')
                    image = Image.open(io.BytesIO(img))
                    image.save(target)
            except:
                # print("invalid url.")
                continue
            qdar.set_postfix(progress=str(inner_ind)+'/'+str(outer_ind)+'/'+str(length), valid2all=str(valid_ind)+'/'+str(index), pid=pid)
        cmd = 'rm -r %s' % (save_dir+'.proc')
        os.system(cmd)
        # todo: save image_list to tar file. 
    


    numProc = 16

    def sub_process(sub_vid_list,pid):
        index = 0
        valid_ind = 0

        qdar = tqdm.tqdm(range(len(sub_vid_list)),total=len(sub_vid_list),ascii=True,position=pid)
        
        for i in qdar:

            wnid = sub_vid_list[i]
            
            save_dir = os.path.join(scratch_dir, wnid) # ../images/wnid/

            if os.path.exists(save_dir):
                continue
            os.makedirs(save_dir)
            wnid_url_list = get_wnid_url_addr(wnid)
            download_pack_save(wnid_url_list,save_dir,index,valid_ind,pid,qdar)

    part0 = int(len(vid_list)/2)
    part1 = int(len(vid_list))
    numIters = int(part0/numProc) + 1
    
    for pid in range(numProc):
        p = Process(target=sub_process, args=(vid_list[int(pid*numIters):min(int((pid+1)*numIters),part0)],pid))
        p.start()
        procList.append(p)
    
    for pid in range(numProc):
        procList[pid].join()


def make_image_list(list_file, image_dir, name, offset=1000):
    with open(list_file) as fp:
        wnid_list = [line.strip() for line in fp]

    save_file = os.path.join(data_dir, 'list', 'img-%s.txt' % name)
    wr_fp = open(save_file, 'w')
    for i, wnid in enumerate(wnid_list):
        img_list = glob.glob(os.path.join(image_dir, wnid, '*.JPEG'))
        for path in img_list:
            index = os.path.join(wnid, os.path.basename(path))
            l = i + offset
            wr_fp.write('%s %d\n' % (index, l))
        if len(img_list) == 0:
            print('Warning: does not have class %s. Do you forgot to download the picture??' % wnid)
    wr_fp.close()


def rm_empty(vid_file):
    with open(vid_file) as fp:
        vid_list = [line.strip() for line in fp]
    cnt = 0
    for i in range(len(vid_list)):
        save_dir = os.path.join(scratch_dir, vid_list[i])
        jpg_list = glob.glob(save_dir + '/*.JPEG')
        if len(jpg_list) < 10:
            print(vid_list[i])
            cmd = 'rm -r %s ' % save_dir
            os.system(cmd)
            cnt += 1
    print(cnt)

def down_sample(list_file, image_dir, size=256):
    with open(list_file) as fp:
        index_list = [line.split()[0] for line in fp]
    for i, index in enumerate(index_list):
        img_file = os.path.join(image_dir, index)
        if not os.path.exists(img_file):
            print('not exist:', img_file)
            continue
        img = downsample_image(img_file, size)
        if img is None:
            continue
        save_file = os.path.join(os.path.dirname(img_file), os.path.basename(img_file).split('.')[0] + 'copy') + '.JPEG'
        cv2.imwrite(save_file, img)
        cmd = 'mv %s %s' % (save_file, img_file)
        os.system(cmd)
        if i % 1000 == 0:
            print(i, len(index_list), index)

def downsample_image(img_file, target_size):
    img = cv2.imread(img_file)
    if img is None:
        return img
    im_shape = img.shape
    im_size_min = np.min(im_shape[0:2])
    im_scale = float(target_size) / float(im_size_min)
    im_scale = min(1, im_scale)
    img = cv2.resize(img, None, None, fx=im_scale, fy=im_scale,
                    interpolation=cv2.INTER_LINEAR)
    return img


def parse_arg():
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--hop', type=str, default='2',
                        help='choice of test difficulties: 2,3,all')
    parser.add_argument('--save_dir', type=str, default=None,
                        help='path to save images')
    parser.add_argument('--user', type=str,
                        help='your username', required=True)
    parser.add_argument('--key', type=str,
                        help='your access key', required=True)
    args = parser.parse_args()
    if args.save_dir is None:
        print('Please set directory to save images')
    return args


args = parse_arg()
scratch_dir = args.save_dir

if __name__ == '__main__':
    if args.hop == '2':
        name = '2-hops'
        list_file = os.path.join(data_dir, 'list/2-hops.txt')
    elif args.hop == '3':
        name = '3-hops'
        list_file = os.path.join(data_dir, 'list/3-hops.txt')
    elif args.hop == 'all':
        name = 'all'
        list_file = os.path.join(data_dir, 'list/all.txt')
    else:
        raise NotImplementedError
    download(list_file)

    make_image_list(list_file, args.save_dir, name)
#     img_file = os.path.join(data_dir, 'list/img-all.txt')
#     down_sample(img_file, args.save_dir)
