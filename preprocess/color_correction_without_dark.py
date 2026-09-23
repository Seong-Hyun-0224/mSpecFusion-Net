from utils.helper import splitFileName, saveImage
# import matplotlib.pyplot as plt
import matplotlib.image as img
import glob
import numpy as np
import cv2

def cropMid(img, size):
    # Crop img
    if len(img.shape) == 3:
        h, w, z = img.shape
    else :
        h, w = img.shape

    mid = int(h/2)
    mid_unit = int(size/2)

    img_crop = img[mid - mid_unit:mid+mid_unit, mid - mid_unit:mid+mid_unit, :]
    return img_crop

## white target 고정 후, 원본 데이터 모두 correct > 결과물 확인
# 2 cal data > save them
import os

def load_ref_white():
    path =  '/Users/dion/workspace/trial/data/ref/test_ref'
    file_ =  '{}_/ref_white.npy'.format(path)
    if not(os.path.exists(file_)):
        files = glob.glob('{}/*'.format(path))
        files = sorted(files)
        print(files)

        list_white = []
        for file in files:
            path, name, _ = splitFileName(file)
            img_white = img.imread(file.replace('test_ref', 'white'))

            img_w_crop = cropMid(img_white, 1024)
            
            img_w_crop = img_w_crop.astype(np.float16)

            img_ = img_w_crop

            max_ch0 = np.max(img_[:,:,0])
            max_ch1 = np.max(img_[:,:,1])
            max_ch2 = np.max(img_[:,:,2])

            # BGR
            ch0_cal = max_ch0 / img_[:,:,0]
            ch1_cal = max_ch1 / img_[:,:,1]
            ch2_cal = max_ch2 / img_[:,:,2]

            ch0_cal = ch0_cal/ np.max(ch0_cal)
            ch1_cal = ch1_cal/ np.max(ch1_cal)
            ch2_cal = ch2_cal/ np.max(ch2_cal)

            h, w = ch0_cal.shape

            img_cal_fin = np.ndarray((h,w,3)).astype(np.float16)
            img_cal_fin[:,:,0] = ch0_cal
            img_cal_fin[:,:,1] = ch1_cal
            img_cal_fin[:,:,2] = ch2_cal

            # image_save = (img_cal_fin*255).astype(np.uint8)
            # fix_img = cv2.cvtColor(image_save, cv2.COLOR_BGR2RGB)
            # saveImage(path_dir_out_cal, name, fix_img)

            list_white.append(img_cal_fin)
        
        ref_white = np.array(list_white)
        np.save(file_, ref_white)
        
    else :
        ref_white = np.load(file_, allow_pickle=True)
        print('load white target success')
    return ref_white


if __name__ == "__main__":
    ref_white = load_ref_white()
    calib = False

    # path =  '/Users/dion/workspace/trial/data/dgist/format_split'
    path =  '/Users/dion/workspace/trial/data/dgist/origin'
    diseases = glob.glob('{}/*'.format(path))
    # print(diseases)

    for disease in diseases:
        cases = glob.glob('{}/*'.format(disease))
        cases = sorted(cases)
        for case in cases:
            chs = glob.glob('{}/*'.format(case))
            chs = sorted(chs)
            print(case)
            for idx, ch in enumerate(chs):
                path_, name, ext = splitFileName(ch)
                path_dir_out = path_.replace('origin', 'origin_')

                img_ = img.imread(ch)

                img_ = cropMid(img_, 1024)

                if calib == True:


                    img_ = img_/255

                    if idx != 9:
                        img_or_cal = img_ * ref_white[idx] * 255 * 1.5
                    else :
                        img_or_cal = img_ *255 * 1.2
                    
                    # img_o = cv2.cvtColor((img_*255).astype(np.uint8), cv2.COLOR_BGR2RGB)
                    # saveImage(path_dir_out, '{}_o'.format(name), img_o)
                    # img_w = cv2.cvtColor((ref_white[idx]*255).astype(np.uint8), cv2.COLOR_BGR2RGB)
                    # saveImage(path_dir_out, '{}_w'.format(name), img_w)

                    image_save = (img_or_cal).astype(np.uint8)
                    fix_img = cv2.cvtColor(image_save, cv2.COLOR_BGR2RGB)
                    saveImage(path_dir_out, name, fix_img)
                else :
                    image_save = (img_).astype(np.uint8)
                    fix_img = cv2.cvtColor(image_save, cv2.COLOR_BGR2RGB)
                    saveImage(path_dir_out, name, fix_img)