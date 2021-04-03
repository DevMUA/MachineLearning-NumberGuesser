from PIL import Image
import numpy as np
import os, os.path


def run():
    folder0 = "trainingdata/0"
    folder1 = "trainingdata/1"
    folder2 = "trainingdata/2"
    folder3 = "trainingdata/3"
    folder4 = "trainingdata/4"
    folder5 = "trainingdata/5"
    folder6 = "trainingdata/6"
    folder7 = "trainingdata/7"
    folder8 = "trainingdata/8"
    folder9 = "trainingdata/9"
    dir : str = "trainingdata/"
    X_matrix = np.empty((0,1680), int)
    Y_matrix = np.empty((1,0),int)
    #getting folder images
    for i in range(10):
        new_dir : str = dir + str(i)
        dir_length : int = len([name for name in os.listdir(folder0) if os.path.isfile(os.path.join(folder0,name))])
        for j in range(dir_length):
            print("analysing training set image")
            filename : str = new_dir + "/" + str(j) + ".png"
            img = Image.open(filename,'r')
            pixel_val = np.array(img.getdata())
            pixel_val.flatten('F')
            img_array = []
            for u in range(len(pixel_val)):
                if pixel_val[u][0] == 255 and pixel_val[u][1] == 255 and pixel_val[u][2] == 255:
                    img_array.append(0)
                else:
                    img_array.append(1)
            X_matrix = np.append(X_matrix,np.array([img_array]), axis=0)
            y_val = []
            helper_int = 0
            if(i==0):
                helper_int = 10
            y_val.append(i+helper_int)
            Y_matrix = np.append(Y_matrix,np.array([y_val]), axis=1)

    Y_matrix = Y_matrix.transpose()
    return X_matrix,Y_matrix

def analyseImage():
    print("Analysing prediction image")
    img = Image.open("prediction/pred.png",'r')
    X_matrix = np.empty((0,1680), int)
    pixel_val = np.array(img.getdata())
    pixel_val.flatten('F')
    img_array = []
    for u in range(len(pixel_val)):
        if pixel_val[u][0] == 255 and pixel_val[u][1] == 255 and pixel_val[u][2] == 255:
            img_array.append(0)
        else:
            img_array.append(1)
    X_matrix = np.append(X_matrix,np.array([img_array]), axis=0)
    return X_matrix

