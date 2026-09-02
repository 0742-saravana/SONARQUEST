import cv2
import numpy as np
import onnxruntime as ort

class YOLOOonnx:
    def __init__(self, model_path):
        self.session = ort.InferenceSession(model_path, providers=['CPUExecutionProvider'])
        self.input_name = self.session.get_inputs()[0].name
        self.input_shape = self.session.get_inputs()[0].shape
        self.input_h = self.input_shape[2]
        self.input_w = self.input_shape[3]

    def predict(self, img_bgr, conf_thresh=0.25, iou_thresh=0.45):
        h, w = img_bgr.shape[:2]
        max_dim = max(h, w)
        padded = np.zeros((max_dim, max_dim, 3), dtype=np.uint8)
        padded[:] = 114
        padded[0:h, 0:w, :] = img_bgr
        
        img_resized = cv2.resize(padded, (self.input_w, self.input_h))
        blob = cv2.dnn.blobFromImage(img_resized, 1/255.0, (self.input_w, self.input_h), swapRB=True, crop=False)

        preds = self.session.run(None, {self.input_name: blob})[0]
        preds = preds[0].transpose(1, 0)

        boxes, scores, class_ids = [], [], []

        for pred in preds:
            confidences = pred[4:]
            class_id = np.argmax(confidences)
            max_conf = confidences[class_id]

            if max_conf > conf_thresh:
                cx, cy, bw, bh = pred[0:4]
                x = cx - bw/2
                y = cy - bh/2
                boxes.append([float(x), float(y), float(bw), float(bh)])
                scores.append(float(max_conf))
                class_ids.append(int(class_id))

        indices = cv2.dnn.NMSBoxes(boxes, scores, conf_thresh, iou_thresh)
        final_dets = []
        scale_ratio = max_dim / float(self.input_w)
        
        if len(indices) > 0:
            for i in indices.flatten():
                box = boxes[i]
                orig_cx = (box[0] + box[2]/2) * scale_ratio
                orig_cy = (box[1] + box[3]/2) * scale_ratio
                orig_w = box[2] * scale_ratio
                orig_h = box[3] * scale_ratio
                
                final_dets.append({
                    "cls": class_ids[i], "conf": scores[i],
                    "cx": orig_cx, "cy": orig_cy, "w": orig_w, "h": orig_h,
                    "box": [orig_cx - orig_w/2, orig_cy - orig_h/2, orig_cx + orig_w/2, orig_cy + orig_h/2]
                })
            
        return final_dets

    def plot(self, img_bgr, dets, class_map):
        annotated = img_bgr.copy()
        for d in dets:
            x1, y1, x2, y2 = [int(v) for v in d["box"]]
            conf = d["conf"]
            label = class_map.get(d["cls"], {}).get("label", "Object")
            cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(annotated, f"{label} {conf:.2f}", (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        return annotated
