import tensorrt as trt
import pycuda.autoinit
import pycuda.driver as cuda
import matplotlib.pyplot as plt

import numpy as np
import cv2
import time
import os
import argparse
import os

import threading
from queue import Queue

from utils.power import check_power, shutdown
from utils.utils import color_box

import PySimpleGUI as sg

import ctypes

import nanocamera as nano # For usb webcams (for testing)


_COLORS = color_box(14).astype(np.float32).reshape(-1, 3)

class VideoWriterThread(threading.Thread):
    def __init__(self, queue, save_path, fps, frame_size):
        super(VideoWriterThread, self).__init__()
        self.queue = queue
        self.vid_writer = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, frame_size)

    def run(self):
        while True:
            item = self.queue.get()
            if item is None:
                break

            frame = item
            self.vid_writer.write(frame)

            self.queue.task_done()

        self.vid_writer.release()

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
    
    def detect_video(self, pipeline, window, end2end=False):
        cv2.namedWindow("FLIR IR Video", cv2.WND_PROP_FULLSCREEN)
        cv2.setWindowProperty("FLIR IR Video", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN) 
        if pipeline == 0:
            cap = cv2.VideoCapture(pipeline, cv2.CAP_GSTREAMER)
        else:
            cap = cv2.VideoCapture(pipeline)
        # cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 512)
        # cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_CONVERT_RGB, 0)
        fps = cap.get(cv2.CAP_PROP_FPS)
        print(fps)
        # w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        # h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        # cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter.fourcc('Y','1','6',' '))
        
        # frame_queue = Queue()
        # video_writer_thread = VideoWriterThread(frame_queue, save_path, 9, (w, h))
        # video_writer_thread.start()

        raw_fps = 9 

        power = True
        pwr_cnt = 0

        conf = self.conf
        infer = True

        while True: 
            screen_width, screen_height = 1920, 1600
            ret, frame = cap.read()

            frame_raw = cv2.bitwise_not(frame) # TEMPORARY: Black hot to white hot. Can change from GUI? <-- need to find out 
            frame = frame_raw

            if not ret:
                break

            blob, ratio = preproc(frame, self.imgsz, self.mean, self.std)
            t1 = time.time()
            data = self.infer(blob)
            
            if infer:
                if end2end: # Using End-to-End engine
                    num, final_boxes, final_scores, final_cls_inds = data
                    final_boxes = np.reshape(final_boxes/ratio, (-1, 4))
                    dets = np.concatenate([final_boxes[:num[0]], np.array(final_scores)[:num[0]].reshape(-1, 1), np.array(final_cls_inds)[:num[0]].reshape(-1, 1)], axis=-1)
                else: # nms not included engine
                    predictions = np.reshape(data, (1, -1, int(5+self.n_classes)))[0]
                    dets = self.postprocess(predictions, ratio)

                if dets is not None:
                    final_boxes, final_scores, final_cls_inds = dets[:, :4], dets[:, 4], dets[:, 5]
                    print(final_cls_inds)
                    frame = vis(frame, final_boxes, final_scores, final_cls_inds, conf=conf, class_names=self.class_names)

            # frame_queue.put(frame)
            calc_fps = int(1000 / ((time.time() - t1) * 1000))

            # frame = cv2.putText(frame, f"FPS:{calc_fps}", (0, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)


            frame_height, frame_width, _ = frame.shape

            scaleWidth = float(screen_width)/float(frame_width)
            scaleHeight = float(screen_height)/float(frame_height)

            if scaleHeight>scaleWidth:
                imgScale = scaleWidth

            else:
                imgScale = scaleHeight

            newX,newY = frame.shape[1]*imgScale, frame.shape[0]*imgScale
            frame = cv2.resize(frame,(int(newX),int(newY)))
            # imgbytes = cv2.imencode('.png', frame)[1].tobytes()
            # img_elem.Update(data=imgbytes)
            # imgbytes = cv2.imencode('.ppm', frame)[1].tobytes()
            # img_elem.draw_image(data=imgbytes, location=(0, 1080))
            cv2.imshow('FLIR IR Video', frame)

            event, values = window.read(timeout=0)
            conf[0] = values['_PEOPLE_']
            conf[2] = values['_CAR_']
            infer_temp = values['_INFER_']
            if infer_temp != infer:
                if infer_temp == True:
                    sg.popup_timed('Inference Mode', button_type=5, auto_close=True, auto_close_duration=3, non_blocking=True, no_titlebar=True, background_color='green', text_color='white')
                else:
                    sg.popup_timed('Raw Video Mode', button_type=5, auto_close=True, auto_close_duration=3, non_blocking=True, no_titlebar=True, background_color='green', text_color='white')
                infer = infer_temp 
            window.TKroot.wm_attributes("-topmost", True)
            if cv2.waitKey(1) & 0xFF == ord('q') or event == 'Exit' or event == None:
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
    # conf_all = conf
    for i in range(len(boxes)):
        box = boxes[i]
        cls_id = int(cls_ids[i])
        score = scores[i]
        # if cls_id == 0:
        #     conf = 0.1
        # else:
        #     conf = conf_all

        if score < conf[cls_id]:
            continue

        x0 = int(box[0])
        y0 = int(box[1])
        x1 = int(box[2])
        y1 = int(box[3])

        color = (_COLORS[cls_id] * 255).astype(np.uint8).tolist()
        # text = '{}:{:.1f}%'.format(class_names[cls_id], score * 100)
        # txt_color = (0, 0, 0) if np.mean(_COLORS[cls_id]) > 0.5 else (255, 255, 255)

        # txt_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.4, 1)[0]
        # cv2.rectangle(img, (x0, y0), (x1, y1), color, 2)

        # txt_bk_color = (_COLORS[cls_id] * 255 * 0.7).astype(np.uint8).tolist()
        # cv2.rectangle(img, (x0, y0 + 1), (x0 + txt_size[0] + 1, y0 + int(1.5 * txt_size[1])), txt_bk_color, -1)
        # cv2.putText(img, text, (x0, y0 + txt_size[1]), cv2.FONT_HERSHEY_SIMPLEX, 0.4, txt_color, thickness=1)
        bracket_thickness = 2
        bracket_length = min(int(0.2 * (y1 - y0)), int(0.3 * (x1 - x0)))

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
        
        self.conf = {
            0:0.4,
            1:0.65,
            2:0.65,
            3:0.65,
            4:0.65,
            5:0.65,
            6:0.65,
            7:0.65,
            8:0.65,
            9:0.65
        }

if __name__ == '__main__':
    sg.theme('DarkGrey4')
    sg.set_options(font=('Arial Bold', 16))
    layout = [
                # [sg.Graph((1920, 1080), (0,0), (1920, 1080), key='_IMAGE_')],
                # [sg.Image(data=None, size=(640, 512), key='_IMAGE_')],
                # [sg.Radio("Inference", "infer", key='_INFER_', default=True)],
                [sg.Checkbox('Inference', default=True, size=(15, 15), key='_INFER_')],
                [sg.Text('People'),
                sg.Slider(range=(0.01, 1), orientation='h', resolution=0.01, default_value=0.4, size=(20, 20), key='_PEOPLE_')],
                [sg.Text('Car'),
                sg.Slider(range=(0.01, 1), orientation='h', resolution=0.01, default_value=0.65, size=(20, 20), key='_CAR_')],
                # [sg.Button('Fullscreen', size=(8, 1), key='_FULL_')],
                [sg.Exit()]
            ]


    parser = argparse.ArgumentParser()
    parser.add_argument("-e", "--engine", required=True, help="TRT engine (from weights directory)")
    parser.add_argument("--end2end", default=False, action="store_true", help="use end2end engine")
    args = parser.parse_args()
    print(args)

    # sg.popup_notify('Loading ADAS')
    sg.popup_timed('Launching IR-ADAS Software', button_type=5, auto_close=True, auto_close_duration=10, no_titlebar=True, background_color='red', text_color='white')
    engine_path = f'/home/jetson/ir-adas/IR-ADAS-FLIR-Jetson/jetson_nano/engines/{args.engine}'

    window = sg.Window('IR-ADAS Development Console', layout, size = (360, 300) , default_element_size=(14, 1), text_justification='left', auto_size_text=False, keep_on_top=True, finalize=True)
    # img_elem = window['_IMAGE_'] 

    pred = Detector(engine_path=engine_path)

    # pipeline = "v4l2src device=/dev/video0 ! video/x-raw,format=(string)I420, interlace-mode=(string)progressive, framerate=30/1 ! videoconvert ! appsink"
    pipeline = 0
    # #pipeline = "v4l2src device=/dev/video0 ! video/x-raw, width=640, height=512, format=(string)I420, pixel-aspect-ratio=1/1, interlace-mode=(string)progressive, framerate=60/1 ! videoconvert ! appsink"
    pred.detect_video(pipeline, window, end2end=args.end2end)
    # video_path = "test3_3min.mp4"
    # pred.detect_video(video_path, window, end2end=args.end2end)
