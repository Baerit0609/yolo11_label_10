# Copyright (C) CVAT.ai Corporation
#
# SPDX-License-Identifier: MIT

import cv2
import numpy as np
import onnxruntime as ort

# -----------------------------------------------------------------------------
# NumPy를 이용한 NMS 함수 구현
# -----------------------------------------------------------------------------
def nms_numpy(boxes, scores, iou_threshold=0.5):
    """
    Args:
        boxes (np.ndarray): 예측된 박스 배열, shape: (N, 4), 각 박스는 [x1, y1, x2, y2] 형식
        scores (np.ndarray): 각 박스의 신뢰도, shape: (N,)
        iou_threshold (float): IoU 임계값 (예: 0.5)
    Returns:
        keep (list): NMS 후 선택된 박스의 인덱스 목록
    """
    indices = np.argsort(scores)[::-1]
    keep = []

    while len(indices) > 0:
        current = indices[0]
        keep.append(current)
        if len(indices) == 1:
            break

        current_box = boxes[current]
        rest_boxes = boxes[indices[1:]]
        xx1 = np.maximum(current_box[0], rest_boxes[:, 0])
        yy1 = np.maximum(current_box[1], rest_boxes[:, 1])
        xx2 = np.minimum(current_box[2], rest_boxes[:, 2])
        yy2 = np.minimum(current_box[3], rest_boxes[:, 3])
        w = np.maximum(0, xx2 - xx1 + 1)
        h = np.maximum(0, yy2 - yy1 + 1)
        inter_area = w * h

        area_current = (current_box[2] - current_box[0] + 1) * (current_box[3] - current_box[1] + 1)
        areas = (rest_boxes[:, 2] - rest_boxes[:, 0] + 1) * (rest_boxes[:, 3] - rest_boxes[:, 1] + 1)
        union_area = area_current + areas - inter_area
        iou = inter_area / union_area

        indices = indices[1:][iou < iou_threshold]

    return keep

# -----------------------------------------------------------------------------
# ModelHandler 클래스 (ONNX 모델 로드 및 추론, NMS 후처리 적용)
# -----------------------------------------------------------------------------
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
        # 이미지 크기를 조정하고 패딩을 추가하여 stride 제약을 만족하도록 함
        shape = im.shape[:2]  # 현재 이미지 높이, 너비
        if isinstance(new_shape, int):
            new_shape = (new_shape, new_shape)

        # Scale ratio 계산 (new / old)
        r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
        if not scaleup:  # 스케일 업은 하지 않음
            r = min(r, 1.0)

        # Resize 후 패딩 크기 계산
        new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
        dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]  # 너비, 높이 패딩

        if auto:  # 최소 직사각형 패딩
            dw, dh = np.mod(dw, stride), np.mod(dh, stride)

        dw /= 2  # 좌우 패딩 분할
        dh /= 2

        if shape[::-1] != new_unpad:  # 이미지 리사이즈
            im = cv2.resize(im, new_unpad, interpolation=cv2.INTER_LINEAR)
        top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
        left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
        im = cv2.copyMakeBorder(im, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)
        return im, r, (dw, dh)

    def _infer(self, inputs: np.ndarray, iou_threshold=0.5):
        try:
            # 전처리: 이미지 색상 변환 및 letterbox 적용
            img = cv2.cvtColor(inputs, cv2.COLOR_BGR2RGB)
            image = img.copy()
            image, ratio, dwdh = self.letterbox(image, auto=False)
            image = image.transpose((2, 0, 1))
            image = np.expand_dims(image, 0)
            image = np.ascontiguousarray(image)

            im = image.astype(np.float32) / 255.0

            inp = {self.input_details[0]: im}
            # ONNX 추론 실행
            detections = self.model.run(self.output_details, inp)[0]

            # 출력 형태에 따라 처리:
            # 만약 detections shape가 (1, N, 6)라면 (N, 6)으로 변경
            if len(detections.shape) == 3:
                detections = detections[0]
            elif len(detections.shape) == 2:
                # YOLOv7 형식: (N, 7) -> 박스 정보가 1~4열, 클래스와 점수가 5열, 마지막 열에 있음
                boxes = detections[:, 1:5].copy()
                class_ids = detections[:, 5].copy()
                scores = detections[:, -1].copy()

                dw, dh = dwdh
                boxes[:, 0] -= dw
                boxes[:, 2] -= dw
                boxes[:, 1] -= dh
                boxes[:, 3] -= dh

                boxes = boxes / ratio
                boxes = boxes.round().astype(np.int32)

                # NMS 적용
                keep_indices = nms_numpy(boxes, scores, iou_threshold=iou_threshold)
                boxes = boxes[keep_indices]
                scores = scores[keep_indices]
                class_ids = class_ids[keep_indices]
                return [boxes, class_ids, scores]

            # 여기서는 detections가 (N, 6) 형태로 가정
            boxes = detections[:, :4].copy()
            scores = detections[:, 4].copy()
            class_ids = detections[:, 5].copy()

            # 패딩 제거 및 원본 이미지 스케일로 복원
            dw, dh = dwdh
            boxes[:, 0] -= dw
            boxes[:, 2] -= dw
            boxes[:, 1] -= dh
            boxes[:, 3] -= dh
            boxes = boxes / ratio
            boxes = boxes.round().astype(np.int32)

            # NMS 후처리 적용
            keep_indices = nms_numpy(boxes, scores, iou_threshold=iou_threshold)
            boxes = boxes[keep_indices]
            scores = scores[keep_indices]
            class_ids = class_ids[keep_indices]

            return [boxes, class_ids, scores]

        except Exception as e:
            print(f"Inference error: {e}")
            import traceback
            traceback.print_exc()
            return None

    def infer(self, image, threshold):
        image = np.array(image)
        # RGB 이미지를 BGR로 변환 (OpenCV는 BGR 사용)
        image = image[:, :, ::-1].copy()
        h, w, _ = image.shape
        detections = self._infer(image)

        results = []
        if detections:
            boxes = detections[0]
            class_ids = detections[1]
            scores = detections[2]

            for cls_id, score, box in zip(class_ids, scores, boxes):
                if score >= threshold:
                    xtl = max(int(box[0]), 0)
                    ytl = max(int(box[1]), 0)
                    xbr = min(int(box[2]), w)
                    ybr = min(int(box[3]), h)

                    results.append({
                        "confidence": str(score),
                        "label": self.labels.get(int(cls_id), "unknown"),
                        "points": [xtl, ytl, xbr, ybr],
                        "type": "rectangle",
                    })

        return results
