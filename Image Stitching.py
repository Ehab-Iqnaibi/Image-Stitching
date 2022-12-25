import cv2 as cv
import numpy as np
import os


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


