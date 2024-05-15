import cv2
import os
import requests
import numpy as np
import json
import random
import math
import operator
from functools import reduce
from ultralytics import YOLO


model = YOLO('yolov8n-pose.pt')
highContrastingColors = ['rgba(0,255,0,1)',"rgba(245, 66, 66,1)","rgba(245, 141, 66,1)","rgba(245, 182, 66,1)","rgba(245, 230, 66,1)","rgba(209, 245, 66,1)","rgba(108, 245, 66,1)","rgba(66, 245, 111,1)","rgba(66, 245, 156,1)","rgba(66, 245, 206,1)","rgba(66, 120, 245,1)","rgba(170, 66, 245,1)","rgba(221, 66, 245,1)","rgba(245, 66, 206,1)","rgba(66, 245, 161,1)","rgba(245, 66, 108,1)","rgba(0,0,0,1)"]


cls_map =     {0: "nose",
     1: "left_eye",
     2: "right_eye",
     3: "left_ear",
     4: "right_ear",
     5: "left_shoulder",
     6: "right_shoulder",
     7: "left_elbow",
     8: "right_elbow",
     9: "left_wrist",
     10: "right_wrist",
     11: "left_hip",
     12: "right_hip",
     13: "left_knee",
     14: "right_knee",
     15: "left_ankle",
     16: "right_ankle"}





def json_creator(inputs, closed):
    """
    Create JSON data from inputs.
    
    Args:
    - inputs (dict): Dictionary containing annotations.
    - closed (bool): Whether the shape is closed.
    
    Returns:
    - str: JSON formatted data.
    """
    data = []
    count = 1
    for index, input in enumerate(inputs):
        color = random.sample(highContrastingColors, 1)[0]

        json_id = count
        sub_json_data = {}
        sub_json_data['id'] = json_id
        sub_json_data['name'] = json_id
        sub_json_data['color'] = color
        sub_json_data['isClosed'] = closed
        sub_json_data['selectedOptions'] = [{"id": "0", "value": "root"},
                                            {"id": str(random.randint(10, 20)), "value": inputs[input][0]}]
        sub_json_data['confidenceScore'] = round(inputs[input][1]*100)
        points = eval(input)
        if len(points) > 0:
            center = tuple(map(operator.truediv, reduce(lambda x, y: map(operator.add, x, y), points), [len(points)] * 2))
            sorted_coords = sorted(points, key=lambda coord: (-135 - math.degrees(
                math.atan2(*tuple(map(operator.sub, coord, center))[::-1]))) % 360)
        else:
            sorted_coords = []
        vertices = []
        is_first = True
        for vertex in sorted_coords:
            vertex_json = {}
            if is_first:
                vertex_json['id'] = json_id
                vertex_json['name'] = json_id
                is_first = False
            else:
                json_id = count
                vertex_json['id'] = json_id
                vertex_json['name'] = json_id
            vertex_json['x'] = vertex[0]
            vertex_json['y'] = vertex[1]
            vertices.append(vertex_json)
            count += 1

        sub_json_data['vertices'] = vertices
        data.append(sub_json_data)
    return json.dumps(data)

def main():

    folder = "/content/videos"
    for vid in os.listdir(folder):
        path = os.path.join(folder, vid)
        video_name = path[path.rfind("/"):path.rfind(".mp4")]
        cam = cv2.VideoCapture(path)
        currentframe = 0

        try:
            if not os.path.exists('data'):
                os.makedirs('data')
        except OSError:
            print ('Error: Creating directory of data')

        while(True):
            annotation = {}
            ret, frame = cam.read()

            if ret:
                img_path = f"/content/data/{video_name}_{str(currentframe)}.png"

                if currentframe % 5 == 0:
                    cv2.imwrite(img_path, frame)

                    yolo_results = model(img_path)
                    kps = yolo_results[0].keypoints
                    if kps.conf == None:
                        tag = "yolo_pose_no_detections"
                    else:
                        tag = "yolo_pose_detections"

                    if kps.conf != None:
                        for num1, kp in enumerate(kps.xyn.cpu().numpy()):
                            for num2, pt in enumerate(kp):
                                xy = np.array([pt[1], pt[0]]) * kps.orig_shape
                                x = xy[0]
                                y = xy[1]
                                conf = kps.conf[num1][num2].cpu().numpy()
                                annotation[str([(y, x)])] = [cls_map[num2], conf]

                    if len(annotation) <= 17:
                        annotations = json_creator(annotation, False)

                        url = "https://autoai-backend-exjsxe2nda-uc.a.run.app/resource"
                        payload = {
                            'model': '6586cfb9a3ed9c3b85198a48',
                            'status': 'backlog',
                            'csv': 'csv',
                            'label': "frame",
                            'tag': tag,
                            'model_type': 'imageAnnotation',
                            'prediction': "track",
                            'confidence_score': 100,
                            'imageAnnotations': str(annotations)
                        }
                        files = [
                            ("resource", (f'{img_path}', open(img_path, 'rb'), 'image/jpeg'))
                        ]
                        headers = {}
                        response = requests.request("POST", url, headers=headers, data=payload, files=files)
                        print(response.text)
                        print('code: ', response.status_code)
                currentframe += 1
            else:
                break

if __name__ == "__main__":
    main()
