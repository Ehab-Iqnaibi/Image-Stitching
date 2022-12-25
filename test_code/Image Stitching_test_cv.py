import cv2 as cv
import numpy as np
import os

'''
def reduceValHSV(val, factor):
    return factor * (val // factor)
def ColorsHSV(img, factor):
    cv.imshow('img1',img[0])
    cv.imshow('img2',img[1])
    size1 = img[0].shape
    size2 = img[1].shape
    lis_rgb_img=[]
    for x in range(2):
        hsv_img = cv.cvtColor(img[x], cv.COLOR_BGR2HSV)
        h, s, v = cv.split(hsv_img)
        if x==0:
            s=size1
        elif x==1:
            s=size2
        for i in range(s[0]):
            for j in range(s[1]):
                h[i, j] = reduceValHSV(h[i, j], factor)
        hsv_img = cv.merge([h, s, v])
        rgb_img = cv.cvtColor(hsv_img, cv.COLOR_HSV2BGR)
        lis_rgb_img.append(rgb_img)
    edge_img(lis_rgb_img,size1,size2)
'''
def reduceValHSV(val, factor):
    return factor * (val // factor)
def ColorsHSV(img, factor):

    hsv_img = cv.cvtColor(img, cv.COLOR_BGR2HSV)
    h, s, v = cv.split(hsv_img)

    size = img.shape
    list_L = []*size[1]
    list_R = []*size[1]
    list_T = []*size[0]
    list_B = []*size[0]
    #loop1
    i = 0
    for x in range(2):
        for j in range(size[1]):
            h[i, j] = reduceValHSV(h[i, j], factor)
            if i == 0:
                list_L.append([h[i, j], s[i, j], v[i, j]])

            elif i==size[0]-1:
                list_R.append([h[i, j], s[i, j], v[i, j]])
        i=size[0]-1

    # loop2
    j = 0
    for x in range(2):
       for i in range(size[0]):
           h[i, j] = reduceValHSV(h[i, j], factor)
           if j == 0:
              list_T.append([h[i, j], s[i, j], v[i, j]])
           elif j == size[1] - 1:
              list_B.append([h[i, j], s[i, j], v[i, j]])
       j = size[1]-1

    #print(list_L)
    print(list_R)
    #print(list_T)
    #print(list_B)
    edge=np.array([[list_L,list_R, list_T,list_B]])
    return edge
'''
   hsv_img = cv.merge([h, s, v])
   rgb_img = cv.cvtColor(hsv_img, cv.COLOR_HSV2BGR)
   return rgb_img
'''

imgFolder='images'
myFolders=os.listdir(imgFolder)
print(myFolders) #folder in images
for folder in myFolders:
    path=imgFolder+'/'+folder
    #print(path)
    lis_img=[]
    edge_img=np.ones((4,600,3))
    mylist=os.listdir(path)
    print(mylist)
    print(f'number of images: {len(mylist)}')
    r=0

    for imgN in mylist:
        current_img=cv.imread(f'{path}/{imgN}')
        #current_img= cv.resize(current_img,(0,0),None,0.2,0.2)
        lis_img.append(current_img)
        cv.imshow(folder, lis_img[r])
        cv.waitKey(5)
        edge_img[r] =ColorsHSV( lis_img[r], 3)
        print(edge_img[r] .shape)
        r = r + 1
        # cv.imshow('reduced colors', reduced_color_img)

    while (True):
        user_input = input('Do you want to choose another image (y/n): ')
        if user_input.lower() == 'yes':
            continue
        elif user_input.lower() == 'no':
            break
'''

 #img1 = cv.imread('D:/Document-E/Master-courses/Python/HW/test images/Jerusalem01.jpg', cv.IMREAD_COLOR)
 #img2 = cv.imread('D:/Document-E/Master-courses/Python/HW/test images/Jerusalem02.jpg', cv.IMREAD_COLOR)
 size1=img1.shape
 size2=img1.shape
 print(size1)
 print(size2)

    stitcher = cv.Stitcher.create()
    result = stitcher.stitch(lis_img)
    cv.imshow(folder,result)
    cv.waitkey(1)


    #print(len(lis_img))
    stitcher =cv.Stitcher.create()
    (status,result)=stitcher.stitch(lis_img)

    if(status == cv.STITCHER_OK):
        print('Panorama Generated')
    else:
        print('Panorama not Generated ')
'''

'''
#final

from Image_Stitching_Functions import edge_img,panorama2,panorama
import os
import cv2 as cv
import numpy as np
from PIL import Image

def similarites(vec1,vec2):
    #Root Mean Square Error
    sim_sub=np.subtract(vec1, vec2)
    sim_rms = np.sqrt(np.mean(np.power(sim_sub, 2)))
    return sim_rms

def panorama(img,index):
    if index == 0:
        pan_img = cv.hconcat([img[1], img[0]])
    elif index == 1:
        pan_img = cv.hconcat([img[0], img[1]])
    elif index == 2:
        pan_img = cv.vconcat([img[1], img[0]])
    elif index == 3:
        pan_img = cv.vconcat([img[0], img[1]])
    cv.imshow('Panorama', pan_img)

def panorama2(path,imgN,index,size):

    img0= Image.open(f'{path}/{imgN[0]}')
    img1= Image.open(f'{path}/{imgN[1]}')
    img0.show()
    img1.show()
    w1 = size[0]
    h1 = size[1]
    w2 = size[2]
    h2 = size[3]
    # (w1, h1) = img[0].size
    # (w2, h2) = img[1].size
    if index == 2:
        pan_img = Image.new('RGB', (w2 + w1, h2))
        pan_img.paste(img1, (0, 0))
        pan_img.paste(img0, (w2, 0))
    elif index == 3:
        pan_img = Image.new('RGB', (w1 + w2, h1))
        pan_img.paste(img0, (0, 0))
        pan_img.paste(img1, (w1, 0))
    elif index == 0:
        pan_img = Image.new('RGB', (w2, h2 + h1))
        pan_img.paste(img1, (0, 0))
        pan_img.paste(img0, (0, h2))
    elif index == 1:
        pan_img = Image.new('RGB', (w1, h1 + h2))
        pan_img.paste(img0, (0, 0))
        pan_img.paste(img1, (0, h1))
    #pan_img.show()
    return(pan_img)

def edge_img(path, imgN):
    # def edge_img(imgpath,img_path):
    img0=cv.imread(f'{path}/{imgN[0]}')
    img1=cv.imread(f'{path}/{imgN[1]}')
    #cv.imshow('img1', img0)
    #cv.imshow('img2', img1)
    #gray = cv.cvtColor(img[0], cv.COLOR_BGR2GRAY)
    img1_array = np.asarray(cv.cvtColor(img0, cv.COLOR_BGR2GRAY))
    img2_array = np.asarray(cv.cvtColor(img1, cv.COLOR_BGR2GRAY))
    #size1
    (w1, h1, c1) = img0.shape
    #size2
    (w2, h2, c2) = img1.shape
    #size=[]
    size=[w1,h1,w2,h1]
    print(str(w1)+','+ str(h1))
    print(str(w2)+','+ str(h2))
    sim_list=[]
    if w1 == w2:
        #Lift & Right edge of img1
        L1 = img1_array[0, :]
        R1 = img1_array[h1 - 1, :]
        # Lift & Right side of img2
        L2 = img2_array[0, :]
        R2 = img2_array[h2 - 1:, :]

        for y in range(2):
            if y ==0:
                sim = similarites(L1, R2)
            elif y== 1:
                sim = similarites(R1, L2)
            sim_list.append(sim)

    else:
        sim_list.append(256**3)
        sim_list.append(256**3)

    if h1 == h2:
        # Top & Bottom edge of img1
        T1 = img1_array[:, 0]
        B1 = img1_array[:, w1 - 1]

        # Top & Bottom edge of img2
        T2 = img2_array[:, 0]
        B2 = img2_array[:, w2 - 1]

        for y in range(2):
            if y == 0:
                sim = similarites(T1, B2)
            elif y== 1:
                sim = similarites(B1, T2)
            sim_list.append(sim)

    elif(h1!= h2) and (w1 == w2):
        sim_list.append(256**3)
        sim_list.append(256**3)

    if (h1!= h2) and (w1 != w2):
        print('stiching is not ok' )

    similar_edges=min(sim_list)
    similar_index = sim_list.index(similar_edges)
    #panorama2(img, similar_index,size)
    return similar_index,size

def _img_path(num):
    imgFolder = 'images'
    myFolders = os.listdir(imgFolder)
    print(myFolders)  # folder in images 
    folder='image'+str(num) # folder title in myFolders:
    path = imgFolder + '/' + folder
    mylist = os.listdir(path)
    print(mylist)
    print(f'number of images: {len(mylist)}')
    #lis_img = []
    img_title=[]
    for imgN in mylist:
        #current_img = cv.imread(f'{path}/{imgN}')
        # current_img= cv.resize(current_img,(0,0),None,0.2,0.2)
        img_title.append(imgN)
        #lis_img.append(current_img)
        #current_img.show()
        # cv.imshow('reduced colors', reduced_color_img)
    #edge_img(lis_img)
    # ColorsHSV(lis_img, 3)
    return path,img_title

while(True):
    num =int(input('Enter number of folder 1,2,3: '))
    (path,img_path)=_img_path(num)
    (similar_index,size)=edge_img(path,img_path)
    pano_img=panorama2(path,img_path, similar_index, size)
    pano_img.show()

    user_input = input('Do you want to choose another image (y/n): ')
    if user_input.lower() == 'y':
        continue
    elif user_input.lower() == 'n':
        break
'''





