
#导入库
import cv2
import numpy as np

pic = cv2.imread("1.jpg") #读入图片
contrast = 1        #对比度
brightness = 100    #亮度
pic_turn = cv2.addWeighted(pic,contrast,pic,0,brightness)
          #cv2.addWeighted(对象,对比度,对象,对比度)
'''cv2.addWeighted()实现的是图像透明度的改变与图像的叠加'''

#cv2.imshow('turn', pic_turn) #显示图片
cv2.imwrite('turn.jpg', pic_turn)







import cv2
import numpy
pic = cv2.imread("1.jpg")
temp = cv2.GaussianBlur(pic, (7,7), 1.5)
#      cv2.GaussianBlur(图像，卷积核，标准差）
cv2.imshow('pic-mohu', temp)
cv2.imwrite('111.jpg', temp)


import cv2
import numpy
import random #random模块用于生成随机数

pic = cv2.imread("1.jpg")

for i in range(1000):
    pic[random.randint(0, pic.shape[0]-1)][random.randint(0,pic.shape[1]-1)][:]=255
cv2.imshow('pic_noise', pic)
cv2.imwrite('pic_noise.jpg', pic)



import cv2
import numpy as np

pic = cv2.imread("1.jpg") #读入图片

h_pic = cv2.flip(pic, 0)#水平翻转
cv2.imshow('fanzhuan',h_pic)
cv2.imwrite('fanzhuan.jpg', h_pic)





#导入库
import cv2
import numpy as np

pic = cv2.imread("1.jpg") #读入图片
height,width = pic.shape[:2] #获取图片的高和宽
#将图像缩小为原来的0.5倍
pic_zoom = cv2.resize(pic, (width//4,height//4), interpolation=cv2.INTER_CUBIC)
         # cv2.resize(图像变量 ,(宽,高)                , 插值方法)   
            
cv2.imshow('zoom', pic_zoom) #显示图片
cv2.imwrite('suofang.jpg', pic_zoom)
cv2.waitKey(0)  
cv2.destroyAllWindows()













