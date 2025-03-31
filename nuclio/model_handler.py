# Copyright (C) CVAT.ai Corporation
#
# SPDX-License-Identifier: MIT

import cv2
import numpy as np
import onnxruntime as ort


class ModelHandler:
    def __init__(self, labels):
        self.model = None
        self.load_network(model="yolo11x.onnx")
        self.labels = labels

    def load_network(self, model):
        device = ort.get_device()
        cuda = True if device == 'GPU' else False
        try:
            providers = ['CUDAExecutionProvider', 'CPUExecutionProvider'] if cuda else ['CPUExecutionProvider']
            so = ort.SessionOptions()
            so.log_severity_level = 3

            self.model = ort.InferenceSession(model, providers=providers, sess_options=so)
            self.output_details = [i.name for i in self.model.get_outputs()]
            self.input_details = [i.name for i in self.model.get_inputs()]

            self.is_inititated = True
        except Exception as e:
            raise Exception(f"Cannot load model {model}: {e}")

    def letterbox(self, im, new_shape=(640, 640), color=(114, 114, 114), auto=True, scaleup=True, stride=32):
        # Resize and pad image while meeting stride-multiple constraints
        shape = im.shape[:2]  # current shape [height, width]
        if isinstance(new_shape, int):
            new_shape = (new_shape, new_shape)

        # Scale ratio (new / old)
        r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
        if not scaleup:  # only scale down, do not scale up (for better val mAP)
            r = min(r, 1.0)

        # Compute padding
        new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
        dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]  # wh padding

        if auto:  # minimum rectangle
            dw, dh = np.mod(dw, stride), np.mod(dh, stride)  # wh padding

        dw /= 2  # divide padding into 2 sides
        dh /= 2

        if shape[::-1] != new_unpad:  # resize
            im = cv2.resize(im, new_unpad, interpolation=cv2.INTER_LINEAR)
        top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
        left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
        im = cv2.copyMakeBorder(im, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)  # add border
        return im, r, (dw, dh)

    def _infer(self, inputs: np.ndarray):
        try:
            img = cv2.cvtColor(inputs, cv2.COLOR_BGR2RGB)
            image = img.copy()
            image, ratio, dwdh = self.letterbox(image, auto=False)
            image = image.transpose((2, 0, 1))
            image = np.expand_dims(image, 0)
            image = np.ascontiguousarray(image)

            im = image.astype(np.float32)
            im /= 255

            inp = {self.input_details[0]: im}
            # ONNX inference
            output = list()
            detections = self.model.run(self.output_details, inp)[0]

            # YOLOv11-NMS 출력 형식 처리
            if len(detections.shape) == 3 and detections.shape[2] == 6:
                # NMS가 적용된 출력 형식 (1, N, 6)
                detections = detections[0]  # (1, N, 6) -> (N, 6)
                
                # [x1, y1, x2, y2, confidence, class_id] 형식으로 해석
                boxes = detections[:, :4]
                scores = detections[:, 4]
                labels = detections[:, 5]
                
                # 좌표 조정 (패딩 제거 및 스케일링)
                dw, dh = dwdh
                boxes[:, 0] -= dw  # x1
                boxes[:, 2] -= dw  # x2
                boxes[:, 1] -= dh  # y1
                boxes[:, 3] -= dh  # y2
                
                # 원본 이미지 스케일로 변환
                boxes /= ratio
                boxes = boxes.round().astype(np.int32)
                
            elif len(detections.shape) == 2:  # YOLOv7 형식 (N, 7)
                boxes = detections[:, 1:5]
                labels = detections[:, 5]
                scores = detections[:, -1]
                
                # 좌표 조정
                dw, dh = dwdh
                boxes[:, 0] -= dw  # x1
                boxes[:, 2] -= dw  # x2
                boxes[:, 1] -= dh  # y1
                boxes[:, 3] -= dh  # y2
                
                boxes /= ratio
                boxes = boxes.round().astype(np.int32)
            else:
                raise ValueError(f"Unexpected detection shape: {detections.shape}")
            
            output.append(boxes)
            output.append(labels)
            output.append(scores)
            return output

        except Exception as e:
            print(f"Inference error: {e}")
            import traceback
            traceback.print_exc()
            return None

    def infer(self, image, threshold):
        image = np.array(image)
        image = image[:, :, ::-1].copy()  # RGB to BGR
        h, w, _ = image.shape
        detections = self._infer(image)

        results = []
        if detections:
            boxes = detections[0]
            labels = detections[1]
            scores = detections[2]

            for label, score, box in zip(labels, scores, boxes):
                if score >= threshold:
                    xtl = max(int(box[0]), 0)
                    ytl = max(int(box[1]), 0)
                    xbr = min(int(box[2]), w)
                    ybr = min(int(box[3]), h)

                    results.append({
                        "confidence": str(score),
                        "label": self.labels.get(int(label), "unknown"),
                        "points": [xtl, ytl, xbr, ybr],
                        "type": "rectangle",
                    })

        return results