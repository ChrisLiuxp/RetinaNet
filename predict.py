#-------------------------------------#
#       对单张图片进行预测
#-------------------------------------#
from retinanet import RetinaNet
from PIL import Image

retinanet = RetinaNet()

# while True:
#     img = input('Input image filename:')
#     try:
#         image = Image.open(img)
#     except:
#         print('Open Error! Try again!')
#         continue
#     else:
#         r_image = retinanet.detect_image(image)
#         r_image.show()

for i in range(1, 22):
    try:
        image = Image.open('img/%d.jpg'%(i))
    except:
        print('Open Error! Try again!')
    else:
        r_image = retinanet.detect_image_fcos(image)
        r_image.show()

try:
    image = Image.open('img/street.jpg')
except:
    print('Open Error! Try again!')
else:
    r_image = retinanet.detect_image_fcos(image)
    r_image.show()