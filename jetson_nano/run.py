import tensorrt as trt
import pycuda.autoinit
import pycuda.driver as cuda
import matplotlib.pyplot as plt

import numpy as np
import cv2
import time
import os
import argparse

from utils.power import check_power, shutdown
from utils.utils import color_box

import nanocamera as nano # For usb webcams (for testing)

_COLORS = color_box(14).astype(np.float32).reshape(-1, 3)

class BaseEngine(object):
    def __init__(self, engine_path):
        self.mean = None
        self.std = None
        self.n_classes = 0
        self.class_names = []

        logger = trt.Logger(trt.Logger.WARNING)
        logger.min_severity = trt.Logger.Severity.ERROR
        runtime = trt.Runtime(logger)

        trt.init_libnvinfer_plugins(logger,'') # initialize TensorRT plugins
        with open(engine_path, "rb") as f:
            serialized_engine = f.read()
        engine = runtime.deserialize_cuda_engine(serialized_engine)

        self.imgsz = engine.get_binding_shape(0)[2:]
        self.context = engine.create_execution_context()
        self.inputs, self.outputs, self.bindings = [], [], []
        self.stream = cuda.Stream()

        for binding in engine:
            size = trt.volume(engine.get_binding_shape(binding))
            dtype = trt.nptype(engine.get_binding_dtype(binding))
            host_mem = cuda.pagelocked_empty(size, dtype)
            device_mem = cuda.mem_alloc(host_mem.nbytes)
            self.bindings.append(int(device_mem))

            if engine.binding_is_input(binding):
                self.inputs.append({'host': host_mem, 'device': device_mem})
            else:
                self.outputs.append({'host': host_mem, 'device': device_mem})


    def infer(self, img):
        self.inputs[0]['host'] = np.ravel(img)
        # transfer data to the gpu
        for inp in self.inputs:
            cuda.memcpy_htod_async(inp['device'], inp['host'], self.stream)
        # run inference
        self.context.execute_async_v2(
            bindings=self.bindings,
            stream_handle=self.stream.handle)
        # fetch outputs from gpu
        for out in self.outputs:
            cuda.memcpy_dtoh_async(out['host'], out['device'], self.stream)
        # synchronize stream
        self.stream.synchronize()

        data = [out['host'] for out in self.outputs]
        return data

    def detect_video(self, pipeline, end2end=False):
        cap = cv2.VideoCapture(pipeline, cv2.CAP_GSTREAMER)
        raw_fps = int(round(cap.get(cv2.CAP_PROP_FPS)))

        conf = 0.5 # HARD CODED conf threshould 
        power = True
        pwr_cnt = 0

        while True:
            ret, frame_raw = cap.read()

            frame_raw = cv2.bitwise_not(frame_raw) # TEMPORARY: Black hot to white hot. Can change from GUI? <-- need to find out 
            frame = frame_raw

            if not ret:
                break

            blob, ratio = preproc(frame_raw, self.imgsz, self.mean, self.std)
            t1 = time.time()
            data = self.infer(blob)
            

            if end2end: # Using End-to-End engine
                num, final_boxes, final_scores, final_cls_inds = data
                final_boxes = np.reshape(final_boxes/ratio, (-1, 4))
                dets = np.concatenate([final_boxes[:num[0]], np.array(final_scores)[:num[0]].reshape(-1, 1), np.array(final_cls_inds)[:num[0]].reshape(-1, 1)], axis=-1)
            else: # nms not included engine
                predictions = np.reshape(data, (1, -1, int(5+self.n_classes)))[0]
                dets = self.postprocess(predictions, ratio)

            fps = int(1000 / ((time.time() - t1) * 1000))

            if dets is not None:
                final_boxes, final_scores, final_cls_inds = dets[:, :4], dets[:, 4], dets[:, 5]
                frame = vis(frame, final_boxes, final_scores, final_cls_inds, conf=conf, class_names=self.class_names)
            
            view_frame = cv2.putText(frame, "FPS:%d " %fps, (0, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            cv2.imshow('FLIR IR Video', frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            power, pwr_cnt = check_power(power, pwr_cnt, raw_fps)
            if power == False:
                break

        cap.release()
        cv2.destroyAllWindows()
        if power == False:
            shutdown()

        

    @staticmethod
    def postprocess(predictions, ratio):
        boxes = predictions[:, :4]
        scores = predictions[:, 4:5] * predictions[:, 5:]
        boxes_xyxy = np.ones_like(boxes)
        boxes_xyxy[:, 0] = boxes[:, 0] - boxes[:, 2] / 2.
        boxes_xyxy[:, 1] = boxes[:, 1] - boxes[:, 3] / 2.
        boxes_xyxy[:, 2] = boxes[:, 0] + boxes[:, 2] / 2.
        boxes_xyxy[:, 3] = boxes[:, 1] + boxes[:, 3] / 2.
        boxes_xyxy /= ratio
        dets = multiclass_nms(boxes_xyxy, scores, nms_thr=0.65, score_thr=0.1)
        return dets

def nms(boxes, scores, nms_thr):
    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]

    areas = (x2 - x1 + 1) * (y2 - y1 + 1)
    order = scores.argsort()[::-1]

    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])

        w = np.maximum(0.0, xx2 - xx1 + 1)
        h = np.maximum(0.0, yy2 - yy1 + 1)
        inter = w * h
        ovr = inter / (areas[i] + areas[order[1:]] - inter)

        inds = np.where(ovr <= nms_thr)[0]
        order = order[inds + 1]

    return keep


def multiclass_nms(boxes, scores, nms_thr, score_thr):
    final_dets = []
    num_classes = scores.shape[1]
    for cls_ind in range(num_classes):
        cls_scores = scores[:, cls_ind]
        valid_score_mask = cls_scores > score_thr
        if valid_score_mask.sum() == 0:
            continue
        else:
            valid_scores = cls_scores[valid_score_mask]
            valid_boxes = boxes[valid_score_mask]
            keep = nms(valid_boxes, valid_scores, nms_thr)
            if len(keep) > 0:
                cls_inds = np.ones((len(keep), 1)) * cls_ind
                dets = np.concatenate(
                    [valid_boxes[keep], valid_scores[keep, None], cls_inds], 1
                )
                final_dets.append(dets)
    if len(final_dets) == 0:
        return None
    return np.concatenate(final_dets, 0)


def preproc(image, input_size, mean, std, swap=(2, 0, 1)):
    if len(image.shape) == 3:
        padded_img = np.ones((input_size[0], input_size[1], 3)) * 114.0
    else:
        padded_img = np.ones(input_size) * 114.0
    img = np.array(image)
    r = min(input_size[0] / img.shape[0], input_size[1] / img.shape[1])
    resized_img = cv2.resize(img, (int(img.shape[1] * r), int(img.shape[0] * r)), interpolation=cv2.INTER_LINEAR,).astype(np.float32)
    padded_img[: int(img.shape[0] * r), : int(img.shape[1] * r)] = resized_img

    padded_img = padded_img[:, :, ::-1]
    padded_img /= 255.0
    if mean is not None:
        padded_img -= mean
    if std is not None:
        padded_img /= std
    padded_img = padded_img.transpose(swap)
    padded_img = np.ascontiguousarray(padded_img, dtype=np.float32)
    return padded_img, r

def vis(img, boxes, scores, cls_ids, conf, class_names=None):
    for i in range(len(boxes)):
        box = boxes[i]
        cls_id = int(cls_ids[i])
        score = scores[i]

        if score < conf:
            continue

        x0 = int(box[0])
        y0 = int(box[1])
        x1 = int(box[2])
        y1 = int(box[3])

        color = (_COLORS[cls_id] * 255).astype(np.uint8).tolist()

        bracket_thickness = 3
        bracket_length = int(0.2 * (y1 - y0))

        cv2.line(img, (x0, y0), (x0, y1), color, bracket_thickness)
        cv2.line(img, (x0, y0), (x0 + bracket_length, y0), color, bracket_thickness)
        cv2.line(img, (x0, y1), (x0 + bracket_length, y1), color, bracket_thickness)
        cv2.line(img, (x1, y0), (x1, y1), color, bracket_thickness)
        cv2.line(img, (x1, y0), (x1 - bracket_length, y0), color, bracket_thickness)
        cv2.line(img, (x1, y1), (x1 - bracket_length, y1), color, bracket_thickness)

    return img

class Detector(BaseEngine):
    def __init__(self, engine_path):
        super(Detector, self).__init__(engine_path)
        self.n_classes = 10
        self.class_names = [
            'person',
            'bike', 
            'car', 
            'motor', 
            'bus', 
            'train', 
            'truck', 
            'dog', 
            'deer', 
            'other vehicle'
            ]

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-e", "--engine", required=True, help="TRT engine (from weights directory)")
    parser.add_argument("--end2end", default=False, action="store_true", help="use end2end engine")

    args = parser.parse_args()
    print(args)
    engine_path = f'engines/{args.engine}'

    pred = Detector(engine_path=engine_path)

    # pipeline = "v4l2src device=/dev/video0 ! video/x-raw,format=(string)I420, interlace-mode=(string)progressive, framerate=60/1 ! videoconvert ! appsink"
    pipeline = 0
    pred.detect_video(pipeline, end2end=args.end2end)
