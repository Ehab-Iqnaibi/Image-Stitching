import cv2 as cv
import numpy as np
import os

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
    print(str(w1)+','+str(w2))
    print(str(h1)+','+str(h2))
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
    panorama(img, similar_index)


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
    if user_input.lower() == 'y':
        continue
    elif user_input.lower() == 'n':
        break