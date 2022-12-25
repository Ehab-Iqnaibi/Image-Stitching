from PIL import Image
import numpy as np
import os

def reduceValHSV(val, factor):
    return factor * (val // factor)
def ColorsHSV(img, factor):
    size1 = img[0].size
    size1  = img[1].size
    print('w1' + ',' + 'w2')
    print('h1'+',' +'h2')
    img_hsv=[]
    for x in range(2):
        hsv_img = img[x].convert("HSV")
        h, s, v = hsv_img.split()
        for i in range(size1[0]):
            for j in range(size1[1]):
                h[i, j] = reduceValHSV(h[i, j], factor)
        img_hsv[x]=Image.merge("HSV",(h, s, v))

#convert image to array
    img_array1 = np.asarray(img_hsv[0])
    img_array2 = np.asarray(img_hsv[1])

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
        current_img = Image.open(f'{path}/{imgN}')
        # current_img= cv.resize(current_img,(0,0),None,0.2,0.2)
        lis_img.append(current_img)
        current_img.show()
        # cv.imshow('reduced colors', reduced_color_img)
    ColorsHSV(lis_img, 3)
    #return mylist

while(True):
    num =int(input('Enter number of folder 1,2,3: '))
    folders(num)

    user_input = input('Do you want to choose another image (y/n): ')
    if user_input.lower() == 'yes':
        continue
    elif user_input.lower() == 'no':
        break

