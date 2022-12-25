"""
Image Stitching HW with openCV,PIL,NUMPY
Water Coloring
"""
from PIL import Image
from Image_Stitching_Functions import edge_img,panorama2,panorama
import os
def _img_path(num):
    imgFolder = 'images'
    myFolders = os.listdir(imgFolder)
    print(myFolders)  # folder in images
    folder='image'+str(num) # folder title in myFolders:
    path = imgFolder + '/' + folder
    mylist = os.listdir(path)
    print(mylist)
    print(f'number of images: {len(mylist)}')
    img_title=[]
    for imgN in mylist:
        img_title.append(imgN)
    return path,img_title

while(True):
    num =int(input('Enter number of folder 1,2,3: '))
    (path,img_title)=_img_path(num)
    img0 = Image.open(f'{path}/{img_title[0]}')
    img1 = Image.open(f'{path}/{img_title[1]}')

    similar_index=edge_img(img0, img1)
    pano_img=panorama2(img0, img1, similar_index)
    pano_img.show()

    user_input = input('Do you want to choose another image (y/n): ')
    if user_input.lower() == 'y':
        continue
    elif user_input.lower() == 'n':
        break