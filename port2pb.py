from __future__ import division
import tensorflow as tf
import numpy as np
from PIL import Image
import os
from port2tf.utils import *
import cv2
import torch
from port2tf.yolov3 import YOLOV3
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

def postprocess(pred_bbox, test_input_size, org_img_shape):
    conf_thres = 0.4
    pred_bbox = np.array(pred_bbox)
    pred_coor = pred_bbox[:, 0:4]
    pred_conf = pred_bbox[:, 4]
    pred_prob = pred_bbox[:, 5:]
    org_h, org_w = org_img_shape
    resize_ratio = min(1.0 * test_input_size / org_w, 1.0 * test_input_size / org_h)
    dw = (test_input_size - resize_ratio * org_w) / 2
    dh = (test_input_size - resize_ratio * org_h) / 2
    pred_coor[:, 0::2] = 1.0 * (pred_coor[:, 0::2] - dw) / resize_ratio
    pred_coor[:, 1::2] = 1.0 * (pred_coor[:, 1::2] - dh) / resize_ratio

    pred_coor = np.concatenate([np.maximum(pred_coor[:, :2], [0, 0]),
                                np.minimum(pred_coor[:, 2:], [org_w - 1, org_h - 1])], axis=-1)
    invalid_mask = np.logical_or((pred_coor[:, 0] > pred_coor[:, 2]), (pred_coor[:, 1] > pred_coor[:, 3]))
    pred_coor[invalid_mask] = 0
    bboxes_scale = np.sqrt(np.multiply.reduce(pred_coor[:, 2:4] - pred_coor[:, 0:2], axis=-1))
    valid_scale = (0, np.inf)
    scale_mask = np.logical_and((valid_scale[0] < bboxes_scale), (bboxes_scale < valid_scale[1]))

    classes = np.argmax(pred_prob, axis=-1)
    scores = pred_conf * pred_prob[np.arange(len(pred_coor)), classes]
    score_mask = scores > conf_thres
    mask = np.logical_and(scale_mask, score_mask)

    coors = pred_coor[mask]
    scores = scores[mask]
    classes = classes[mask]
    bboxes = np.concatenate([coors, scores[:, np.newaxis], classes[:, np.newaxis]], axis=-1)
    bboxes = nms(bboxes, conf_thres, 0.45, method='nms')
    return bboxes


def freeze_graph(checkpoint_path, output_node_names, savename):
    with tf.name_scope('input'):
        input_data = tf.placeholder(dtype=tf.float32, shape=(1, INPUTSIZE, INPUTSIZE, 3), name='input_data')
        training = tf.placeholder(dtype=tf.bool, name='training')
    statedict=torch.load(checkpoint_path,map_location=torch.device('cpu'))['state_dict']
    print(statedict.keys())
    assert 0
    output = YOLOV3(training).build_network_dynamic(input_data,statedict,inputsize=INPUTSIZE)
    with tf.Session() as sess:
        net_vars = tf.get_collection('YoloV3')
        # net_vars=[var for var in net_vars if var.name not in filterlist]
        saver = tf.train.Saver(net_vars)
        saver.restore(sess, checkpoint_path)
        output_graph_def = tf.graph_util.convert_variables_to_constants(
            sess=sess,
            input_graph_def=sess.graph_def,
            output_node_names=output_node_names.split(","))
        with tf.gfile.GFile('{}/{}'.format('port', savename), "wb") as f:
            f.write(output_graph_def.SerializeToString())
        print("%d ops in the final graph." % len(output_graph_def.node))


def freeze_graph_test(pb_path, outnode):
    with tf.Graph().as_default():
        output_graph_def = tf.GraphDef()
        with open(pb_path, "rb") as f:
            output_graph_def.ParseFromString(f.read())
        for idx, node in enumerate(output_graph_def.node):
            if node.op == 'Conv2D' and 'explicit_paddings' in node.attr:
                del node.attr['explicit_paddings']
            if node.op == 'ResizeNearestNeighbor' and 'half_pixel_centers' in node.attr:
                del node.attr['half_pixel_centers']
        tf.import_graph_def(output_graph_def, name="")
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            img = cv2.imread('assets/people.jpg')
            originimg = img
            orishape = img.shape
            img = img_preprocess2(img, None, (INPUTSIZE, INPUTSIZE), False, keepratio=False)
            img = img.astype(np.float32)[np.newaxis, ...]
            inputdata = sess.graph.get_tensor_by_name("input/input_data:0")
            outbox_flag = sess.graph.get_tensor_by_name('{}:0'.format(outnode))
            outbox = sess.run(outbox_flag, feed_dict={inputdata: img})
            outbox = np.array(postprocess(outbox, INPUTSIZE, orishape[:2]))
            originimg = draw_bbox(originimg, outbox, CLASSES)
            cv2.imshow('w', originimg)
            cv2.waitKey(-1)


if __name__ == '__main__':
    INPUTSIZE = 512
    CLASSES=[
  "aeroplane",
  "bicycle",
  "bird",
  "boat",
  "bottle",
  "bus",
  "car",
  "cat",
  "chair",
  "cow",
  "diningtable",
  "dog",
  "horse",
  "motorbike",
  "person",
  "pottedplant",
  "sheep",
  "sofa",
  "train",
  "tvmonitor"
  ]
    savename = 'coco{}'.format(INPUTSIZE)
    outnodes = "YoloV3/output/boxconcat"
    ckptpath = 'checkpoints/strongerv3_sparse/checkpoint-best-ft0.3.pth'
    freeze_graph(checkpoint_path=ckptpath, output_node_names=outnodes, savename='%s.pb' % savename)
    freeze_graph_test('port/%s.pb' % savename, outnodes)
    # onnx_test()

