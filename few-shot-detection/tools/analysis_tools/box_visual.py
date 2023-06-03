import json
import os
import random

import cv2

root_path = 'data/voc/'
SAMPLE_NUMBER = 4900  # 随机挑选多少个图片检查，
id_category = {1:'aeroplane', 2:'bicycle', 3:'boat', 4:'bottle', 5:'car', 6:'cat',
                        7:'chair', 8:'diningtable', 9:'dog', 10:'horse', 11:'person',
                        12:'pottedplant', 13:'sheep', 14:'train', 15:'tvmonitor', 16:'bird',
                        17:'bus', 18:'cow', 19:'motorbike', 20:'sofa'}


def visiual():
    # 获取bboxes
    json_file = os.path.join(root_path, 'PascalVoc_CocoStyle/annotations/pascal_test2007_split1.json')  # 如果想查看验证集，就改这里
    data = json.load(open(json_file, 'r'))
    images = data['images']  # json中的image列表，

    # 读取图片
    for i in random.sample(images, SAMPLE_NUMBER):  # 随机挑选SAMPLE_NUMBER个检测
        # for i in images:                                        # 整个数据集检查
        img = cv2.imread(os.path.join(root_path, 'PascalVoc_CocoStyle/images',
                                      i['file_name']))  # 改成验证集的话，这里的图片目录也需要改,train2017 -> val2017
        bboxes = []  # 获取每个图片的bboxes
        category_ids = []
        annotations = data['annotations']
        for j in annotations:
            if j['image_id'] == i['id']:
                bboxes.append(j["bbox"])
                category_ids.append(j['category_id'])

        # 生成锚框
        for idx, bbox in enumerate(bboxes):
            left_top = (int(bbox[0]), int(bbox[1]))  # 这里数据集中bbox的含义是，左上角坐标和右下角坐标。
            right_bottom = (int(bbox[0] + bbox[2]), int(bbox[1] + bbox[3]))  # 根据不同数据集中bbox的含义，进行修改。
            cv2.rectangle(img, left_top, right_bottom, (0, 255, 0), 1)  # 图像，左上角，右下坐标，颜色，粗细
            cv2.putText(img, id_category[category_ids[idx]], left_top, cv2.FONT_HERSHEY_SCRIPT_SIMPLEX, 0.4,
                        (255, 255, 255), 1)
            # 画出每个bbox的类别，参数分别是：图片，类别名(str)，坐标，字体，大小，颜色，粗细
        # cv2.imshow('image', img)                                          # 展示图片，
        # cv2.waitKey(1000)
        cv2.imwrite(os.path.join('visiual', i['file_name']), img)  # 或者是保存图片
    # cv2.destroyAllWindows()


if __name__ == '__main__':
    print('—' * 50)
    os.mkdir('visiual')
    visiual()
    print('| visiual completed.')
    print('| saved as ', os.path.join(os.getcwd(), 'visiual'))
    print('—' * 50)
