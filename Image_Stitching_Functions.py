import cv2 as cv
import numpy as np
from PIL import Image

def similarites(vec1,vec2):
    #Root Mean Square Error
    sim_sub=np.subtract(vec1, vec2)
    sim_rms = np.sqrt(np.mean(np.power(sim_sub, 2)))
    return sim_rms
#for open CV
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

def panorama2(img0, img1,index):
    img0.show()
    img1.show()
    # width and height of images
    (w1, h1) = img0.size
    (w2, h2) = img1.size
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

    return(pan_img)

def edge_img(img0, img1):

    # convert image to gray
    img0_array = np.asarray(img0.convert('L'))
    img1_array = np.asarray(img1.convert('L'))

    (w1, h1) = img0.size
    (w2, h2) = img1.size

    print(str(w1)+','+ str(h1))
    print(str(w2)+','+ str(h2))
    sim_list=[]
    if w1 == w2:
        #Lift & Right edge of img1
        L1 = img0_array[0, :]
        R1 = img0_array[h1 - 1, :]
        # Lift & Right side of img2
        L2 = img1_array[0, :]
        R2 = img1_array[h2 - 1:, :]

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
        T1 = img0_array[:, 0]
        B1 = img0_array[:, w1 - 1]

        # Top & Bottom edge of img2
        T2 = img1_array[:, 0]
        B2 = img1_array[:, w2 - 1]

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

    return similar_index
