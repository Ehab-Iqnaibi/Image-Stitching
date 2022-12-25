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
def similarites(vec1,vec2):
    #Root Mean Square Error
    sim_sub=np.subtract(vec1, vec2)
    sim_rms = np.sqrt(np.mean(np.power(sim_sub, 2)))
    return sim_rms


def edge_img(img):
    cv.imshow('img1', img[0])
    cv.imshow('img2', img[1])
    #gray = cv.cvtColor(img[0], cv.COLOR_BGR2GRAY)
    img1_array = np.asarray(cv.cvtColor(img[0], cv.COLOR_BGR2GRAY))
    img2_array = np.asarray(cv.cvtColor(img[1], cv.COLOR_BGR2GRAY))
    #size1
    (w1, h1, c1) = img[0].shape
    #size2
    (w2, h2, c2) = img[1].shape
    #print(str(w1)+','+str(w2))
    #print(str(h1)+','+str(h2))
    sim_list=[]
    if w1 == w2:
        #Lift & Right edge of img1
        L1 = img1_array[0, :]
        R1 = img1_array[height1 - 1, :]
        # Lift & Right side of img2
        L2 = img2_array[0, :]
        R2 = img2_array[h2 - 1:, :]

        for y in range(2):
            if y ==0:
                sim = similarites(L1, R2)
            elif y== 1:
                sim2 = similarites(R1, L2)
            sim_list.append(sim)

    else:
        rms_lis.append(1000000)
        rms_lis.append(1000000)

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

    else:
        rms_lis.append(1000000)
        rms_lis.append(1000000)

    min_rms_pos = rms_lis.index(min(rms_lis))


def folders(num):
    imgFolder = 'images'
    myFolders = os.listdir(imgFolder)
    print(myFolders)  # folder in images
    #for folder in myFolders:
    folder='image'+str(num)
    path = imgFolder + '/' + folder
    mylist = os.listdir(path)
    print(mylist)
    print(f'number of images: {len(mylist)}')
    lis_img = []
    for imgN in mylist:
        current_img = cv.imread(f'{path}/{imgN}')
        # current_img= cv.resize(current_img,(0,0),None,0.2,0.2)
        lis_img.append(current_img)
        #current_img.show()
        # cv.imshow('reduced colors', reduced_color_img)
    edge_img(lis_img)
    #ColorsHSV(lis_img, 3)


while(True):
    num =int(input('Enter number of folder 1,2,3: '))
    folders(num)

    user_input = input('Do you want to choose another image (y/n): ')
    if user_input.lower() == 'yes':
        continue
    elif user_input.lower() == 'no':
        break