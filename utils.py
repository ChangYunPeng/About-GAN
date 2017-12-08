import os
import h5py
import random
import gc
from PIL import Image
import numpy as np
import h5py
import math
import matplotlib.pyplot as plt

os.environ["PATH"] += os.pathsep + 'C:/Program Files (x86)/Graphviz2.38/bin/'

# file_name = '../../../SOURCE/VOC_level_124_RGB_gridesize_80_sig07_bic24_MAXNUM_35000.h5'
file_name = '../../../SOURCE/new_VOC_power2_level_124_gridesize_80_Gaussian_05_no_deborder_MAXNUM_30000_SELECTNUM_15000.h5'

def load_level_data(file_name,group_name):
    f = h5py.File(file_name, 'r')
    data = f[group_name].value
    a1,a2,a3,a4,a5 = np.array(data).shape
    print(a1,a2,a3,a4,a5)
    data = np.reshape(data,(a1,a3,a4,a5))
    my_data=np.array(data,dtype='double')
    my_data = np.transpose(my_data,[0,3,2,1])
    return my_data

def load_data(level = 2):
    data_group_name = 'data_channel_' + str(1) + 'level_' + str(level)
    data = load_level_data(file_name, data_group_name)
    return data

def load_label(level = 1):
    data_group_name = 'label_channel_' + str(1) + 'level_' + str(level)
    data = load_level_data(file_name, data_group_name)
    return data

def make_trainable(net, val):
    net.trainable = val
    for l in net.layers:
        l.trainable = val

def load_rand_data(SELECT_NUM = 5000,my_load_level = 2):
    my_label = load_data(my_load_level)
    a1, a2, a3, a4 = my_label.shape
    DATA_NUM = a1

    if SELECT_NUM > DATA_NUM:
        SELECT_NUM = DATA_NUM

    SELECT_LIST = np.array(random.sample(range(0, DATA_NUM), SELECT_NUM), dtype='int')
    my_label = my_label[SELECT_LIST, :, :, :]
    gc.collect()
    return my_label

def load_rand_label(SELECT_NUM = 5000):
    my_label = load_label(1)
    a1, a2, a3, a4 = my_label.shape
    DATA_NUM = a1

    if SELECT_NUM > DATA_NUM:
        SELECT_NUM = DATA_NUM

    SELECT_LIST = np.array(random.sample(range(0, DATA_NUM), SELECT_NUM), dtype='int')
    my_label = my_label[SELECT_LIST, :, :, :]
    gc.collect()
    return my_label

def test_single_gray_image(model,num = 1,my_level = 2):
    level = my_level
    imag_path = '../../SOURCE/Img/Set14/comic.bmp'
    imag_save_path = '../RESULT/X' +str(my_level) +'/'

    test_Image = Image.open(imag_path)
    test_Image = test_Image.convert('L')
    test_Image.save(imag_save_path+'Original.bmp')

    w, h = test_Image.size
    w_n = int(w / level)
    h_n = int(h / level)
    my_size = w, h
    my_low_size = w_n, h_n

    test_Image = test_Image.resize(my_low_size, Image.BICUBIC)
    test_Image = test_Image.resize(my_size, Image.BICUBIC)

    pixels = np.array(test_Image)
    pixels.astype(np.double)
    pixels = pixels / 255.0

    # mean_pixels = pixels.mean()

    in_put = np.array(pixels,dtype='double')
    a2,a3 = in_put.shape
    in_put = np.reshape(in_put,(1,1,a2,a3))

    output = np.array(model.predict(in_put))
    b1,b2,b3,b4 = output.shape
    output = np.reshape(output,(b3,b4))

    result_image = np.array(output,dtype='double')

    result_image = result_image * 255.0
    result_image = np.ceil(result_image)
    result_image = result_image.astype('uint8')
    print(result_image.dtype)
    img = Image.fromarray(result_image, mode='L')

    test_Image.save(imag_save_path + 'input.bmp')
    img.save(imag_save_path + str(num)+'output.bmp')

def load_rand_data_label(SELECT_NUM = 5000,my_load_level =2):

    my_label = load_label(1)
    a1, a2, a3, a4 = my_label.shape
    DATA_NUM = a1

    SELECT_LIST = np.array(random.sample(range(0, DATA_NUM), SELECT_NUM), dtype='int')
    my_label = my_label[SELECT_LIST, :, :, :]
    gc.collect()

    load_level = int(my_load_level)
    my_data = load_data(load_level)
    my_data = my_data[SELECT_LIST, :, :, :]
    gc.collect()
    return my_data,my_label

def load_rand_data1_data2_label(SELECT_NUM = 5000,my_input_1 =4, my_input_2 = 2):

    my_label = load_label(1)
    a1, a2, a3, a4 = my_label.shape
    DATA_NUM = a1

    SELECT_LIST = np.array(random.sample(range(0, DATA_NUM), SELECT_NUM), dtype='int')
    my_label = my_label[SELECT_LIST, :, :, :]
    gc.collect()

    load_level = int(my_input_1 / 2) + 1
    my_data_1 = load_data(load_level)
    my_data_1 = my_data_1[SELECT_LIST, :, :, :]
    gc.collect()

    load_level = int(my_input_2 / 2) + 1
    my_data_2 = load_data(load_level)
    my_data_2 = my_data_2[SELECT_LIST, :, :, :]
    gc.collect()

    # my_data_1 = np.array(my_data_1, dtype=np.float32)
    # my_data_2 = np.array(my_data_2, dtype=np.float32)
    # my_label = np.array(my_label,dtype=np.float32)
    return my_data_1,my_data_2,my_label

def test_4difference_single_rgb_image(model,num = 1,my_level = 2):
    level = my_level
    imag_path = '../../SOURCE/Img/Set14/comic.bmp'
    imag_save_path = '../RESULT/X' +str(my_level) +'/'

    test_Image = Image.open(imag_path)
    test_Image.save(imag_save_path+'Original.bmp')

    w, h = test_Image.size
    w_n = int(w / level)
    h_n = int(h / level)
    my_size = w, h
    my_low_size = w_n, h_n

    test_Image = test_Image.resize(my_low_size, Image.BICUBIC)
    test_Image = test_Image.resize(my_size, Image.BICUBIC)

    pixels = np.array(test_Image)
    pixels.astype(np.double)
    pixels = pixels / 255.0

    in_put = np.array(np.transpose(pixels,(2,1,0)),dtype='double')
    a1,a2,a3 = in_put.shape
    in_put = np.reshape(in_put,(1,a1,a2,a3))

    output = np.array(model.predict(in_put))
    b1,b2,b3,b4 = output.shape
    output = np.reshape(output,(b2,b3,b4))

    result_image = np.array(np.transpose(output,(2,1,0)),dtype='double')
    result_image = result_image + pixels
    result_image = result_image * 255.0
    result_image = np.ceil(result_image)
    result_image = result_image.astype('uint8')
    print(result_image.dtype)
    img = Image.fromarray(result_image, mode='RGB')

    test_Image.save(imag_save_path + 'input.bmp')
    img.save(imag_save_path + str(num)+'output.bmp')
    return

def test_single_rgb_image(model,num = 1,my_level = 2):
    # cur_model = model.cpu()
    level = my_level
    imag_path = '../../../python_new/SOURCE/Img/Set14/comic.bmp'
    imag_save_path = '../RESULT/X' +str(my_level) +'/'

    test_Image = Image.open(imag_path)
    test_Image.save(imag_save_path+'Original.bmp')

    w, h = test_Image.size
    w_n = int(w / level)
    h_n = int(h / level)
    my_size = w, h
    my_low_size = w_n, h_n

    test_Image = test_Image.resize(my_low_size, Image.BICUBIC)
    test_Image = test_Image.resize(my_size, Image.BICUBIC)

    pixels = np.array(test_Image)
    pixels = pixels[0:80,0:80,:]
    pixels.astype(np.double)
    pixels = pixels / 255.0

    in_put = np.array(np.transpose(pixels,(2,1,0)),dtype=np.float32)
    a1,a2,a3 = in_put.shape
    print in_put.shape
    in_put = np.reshape(in_put,(1,a1,a2,a3))
    in_put = np.concatenate([in_put, in_put],axis=0)
    in_put = Variable(torch.from_numpy(in_put), requires_grad=False)


    if torch.cuda.is_available():
        in_put = in_put.cuda()
    # print in_put
    output = model(in_put)


    # print in_put
    print output
    output = output.cpu()
    output = np.array(output.data.numpy())
    b1,b2,b3,b4 = output.shape
    output = output[0,:,:,:]

    result_image = np.array(np.transpose(output,(2,1,0)),dtype='double')

    result_image = result_image * 255.0
    result_image = np.ceil(result_image)
    result_image = result_image.astype('uint8')
    print(result_image.dtype)
    img = Image.fromarray(result_image, mode='RGB')

    test_Image.save(imag_save_path + 'input.bmp')
    img.save(imag_save_path + str(num)+'output.bmp')

def load_single_rgb_image(my_level = 2):
    level = my_level
    imag_path = '../../../python_new/SOURCE/Img/Set14/comic.bmp'


    test_Image = Image.open(imag_path)

    w, h = test_Image.size
    w_n = int(w / level)
    h_n = int(h / level)
    my_size = w, h
    my_low_size = w_n, h_n

    test_Image = test_Image.resize(my_low_size, Image.BICUBIC)
    test_Image = test_Image.resize(my_size, Image.BICUBIC)

    pixels = np.array(test_Image)
    pixels.astype(np.double)
    pixels = pixels / 255.0

    in_put = np.array(np.transpose(pixels,(2,1,0)),dtype=np.float32)
    a1,a2,a3 = in_put.shape
    in_put = np.reshape(in_put,(1,a1,a2,a3))

    return in_put