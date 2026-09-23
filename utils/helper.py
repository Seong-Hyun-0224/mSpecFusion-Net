import os
import cv2
import glob
import shutil
import numpy as np
from skimage import io
import csv
# import sys
# sys.path.append('/home/project')

def getPath():
    return os.getcwd()

def splitFileName(filename):
    if len(filename.split('.')) > 2:
        if filename[0] == '.':
            filename = filename[1:]
            path = os.path.dirname(filename)
            name, extension = os.path.basename(filename).split('.')
        else :
            path = os.path.dirname(filename)
            name = os.path.basename(filename)[:-8]
            extension = os.path.basename(filename)[-7:]
    else :
        path = os.path.dirname(filename)
        name, extension = os.path.basename(filename).split('.')
    return path, name, extension

def makedirs(dir):
    '''
        dir - whole path
    '''
    os.makedirs('{}'.format(dir), exist_ok=True)
    os.chmod('{}'.format(os.path.dirname(dir)), 0o777)
    os.chmod('{}'.format(dir), 0o777)
    return

def readImage(file, imgtype='png'):
    if imgtype == 'tif':
        img = io.imread(file)
    else :
        img = cv2.imread(file)
    return img

def saveImage(dir, filename, image, imgtype='png'):
    makedirs(dir)
    if imgtype == 'tif':
        filename = filename.replace('.tif', '')
        io.imsave('{}/{}.tif'.format(dir, filename), image, check_contrast=False)
    else :
        filename = filename.replace('.png', '')
        file = '{}/{}.png'.format(dir,filename)
        cv2.imwrite(file, image)
    return

def cleanDirs(dirs):
    for folder_path in dirs:
        if os.path.exists(folder_path) and os.path.isdir(folder_path):
            try:
                shutil.rmtree(folder_path)
                print(f"폴더가 성공적으로 제거되었습니다: {folder_path}")
            except Exception as e:
                print(f"폴더를 제거하는 동안 오류가 발생했습니다: {e}")
        else:
            print("해당 경로가 없거나 폴더가 아닙니다.")
        makedirs(folder_path)
    return

def getNameList(path_dir, type='*.geojson', path_default = True):
    if path_default == True:
        files = glob.glob('{}/{}/{}'.format(getPath(),path_dir, type))
    else :
        files = glob.glob('{}/{}'.format(path_dir, type))
    files_ = []
    for file in files:
        p, name, e = splitFileName(file)
        # files_.append(name.replace('_result', '')[10:])
        files_.append(name)
    return files_

def cropMid(img, size):
    # Crop img
    if len(img.shape) == 3:
        h, w, z = img.shape
    else :
        h, w = img.shape
    
    if size > h or size > w:
        return []

    mid_h = int(h/2)
    mid_w = int(w/2)
    # mid_unit = int(size/2)
    mid_unit = 256

    if len(img.shape) == 3:
        img_crop = img[mid_h - mid_unit:mid_h+mid_unit, mid_w - mid_unit:mid_w+mid_unit, :]
    else :
        img_crop = img[mid_h - mid_unit:mid_h+mid_unit, mid_w - mid_unit:mid_w+mid_unit]
    
    return img_crop

def writeCSV(row=['nothing'], filename='result/result.csv', flag_append=True):
    # row = [date, time, model, optimizer, dataname, #ch, epoch, loss, acc, f1, auc]
    # batch? elapsedtime?
    # with open('test.csv', 'a') as myfile: - newline default - \n
    if flag_append:
        with open(filename, 'a', newline='') as csvfile:
            wr = csv.writer(csvfile)
            wr.writerow(row)
    
    else :
        with open(filename, 'w', newline='') as csvfile:
            wr = csv.writer(csvfile)
            wr.writerow(row)

import json 
import sys 
import requests 

def sendMSG(code, msg, status):
    """
    Optional Slack notification for long-running jobs.
    Set the SLACK_WEBHOOK_URL environment variable to enable; no-op otherwise.
    """
    url = os.environ.get('SLACK_WEBHOOK_URL')
    if not url:
        return

    title = code
    message = msg

    if status == 'error':
        color = '#FF0000'
    elif status == 'end':
        color = '#000000'
    else:
        color = '#0080ff'

    slack_data = {
        "username": "swBot",
        "icon_emoji": ":satellite:",
        "attachments": [
            {
                "color": color,
                "fields": [
                    {
                        "title": title,
                        "value": message,
                        "short": "false",
                    }
                ]
            }
        ]
    }

    byte_length = str(sys.getsizeof(slack_data))
    headers = {'Content-Type': "appication.json", 'Content-Length' : byte_length}
    response = requests.post(url, data = json.dumps(slack_data), headers = headers)
    if response.status_code != 200:
        raise Exception(response.status_code, response.text)