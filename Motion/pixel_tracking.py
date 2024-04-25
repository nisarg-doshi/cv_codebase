import numpy as np
import cv2
from ultralytics import YOLO

# Need to make it more modular and usable

class PixelTracking:
    def __init__(self, video_source=2, model_path='2Mar_model.pt'):
        self.cap = cv2.VideoCapture(video_source)
        self.model = YOLO(model_path)
        self.color = np.random.randint(0, 255, (100, 3))
        self.feature_params = dict(maxCorners=100, qualityLevel=0.3, minDistance=7, blockSize=7)
        self.lk_params = dict(winSize=(100, 100), maxLevel=5, criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))
        self.num_points = 30

    def generate_points_inside_polygon(self, polygon_coords):
        min_x, min_y = np.min(polygon_coords, axis=0)
        max_x, max_y = np.max(polygon_coords, axis=0)
        random_points = []
        while len(random_points) < self.num_points:
            x = np.random.uniform(min_x, max_x)
            y = np.random.uniform(min_y, max_y)
            if self.is_point_inside_polygon(x, y, polygon_coords):
                random_points.append((x, y))
        return np.array(random_points)

    @staticmethod
    def is_point_inside_polygon(x, y, polygon_coords):
        n = len(polygon_coords)
        inside = False
        p1x, p1y = polygon_coords[0]
        for i in range(1, n + 1):
            p2x, p2y = polygon_coords[i % n]
            if y > min(p1y, p2y) and y <= max(p1y, p2y) and x <= max(p1x, p2x):
                if p1y != p2y:
                    xinters = (y - p1y) * (p2x - p1x) / (p2y - p1y) + p1x
                    if p1x == p2x or x <= xinters:
                        inside = not inside
            p1x, p1y = p2x, p2y
        return inside

    def run(self):
        ret, old_frame = self.cap.read()
        old_gray = cv2.cvtColor(old_frame, cv2.COLOR_BGR2GRAY)
        result = self.model(old_frame)
        polygon_coords = result[0].masks.xy[0]
        p0 = np.array([[list(x)] for x in self.generate_points_inside_polygon(polygon_coords)]).astype(np.float32)
        mask = np.zeros_like(old_frame)
        prev_keys = "None"
        reset_key_pressed = False
        
        while True:
            ret, frame = self.cap.read()
            frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            
            if reset_key_pressed:
                result = self.model(frame)
                polygon_coords = result[0].masks.xy[0]
                p0 = np.array([[list(x)] for x in self.generate_points_inside_polygon(polygon_coords)]).astype(np.float32)
                reset_key_pressed = False
                mask = np.zeros_like(frame)
                old_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                prev_keys = "None"

            p1, st, err = cv2.calcOpticalFlowPyrLK(old_gray, frame_gray, p0, None, **self.lk_params)

            try:
                good_new = p1[st == 1]
                good_old = p0[st == 1]
            except:
                continue
            
            if prev_keys == "None":
                prev_keys = [np.mean([x[0] for x in good_new]), np.mean([x[1] for x in good_new])]
            curr_keys = [np.mean([x[0] for x in good_new]), np.mean([x[1] for x in good_new])]
            
            for i, (new, old) in enumerate(zip(good_new, good_old)):
                a, b = new.ravel()
                c, d = old.ravel()
                a, b, c, d = round(a), round(b), round(c), round(d)
                mask = cv2.line(mask, (a, b), (c, d), self.color[i].tolist(), 2)
                frame = cv2.circle(frame, (a, b), 5, self.color[i].tolist(), -1)
            
            img = cv2.add(frame, mask)
            img = cv2.line(img, [round(curr_keys[0]), round(curr_keys[1])], [round(prev_keys[0]), round(prev_keys[1])], (255, 125, 255), 19)
            img = cv2.polylines(img, np.int32([polygon_coords]), True, (255, 0, 0), 2)
            
            status = "Invalid Motion"
            if prev_keys[0] > curr_keys[0]:
                if abs(prev_keys[0] - curr_keys[0]) > 20:
                    status = "Return"
                else:
                    status = "Invalid Motion"
            if prev_keys[0] < curr_keys[0]:
                if abs(prev_keys[0] - curr_keys[0]) > 20:
                    status = "Grab"
                else:
                    status = "Invalid Motion"

            img = cv2.putText(img, status, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)
            cv2.imshow('frame', img)
            
            k = cv2.waitKey(25)
            if k == 27:
                break
            elif k == ord('r') or k == ord('R'):
                reset_key_pressed = True
            
            old_gray = frame_gray.copy()
            p0 = good_new.reshape(-1, 1, 2)

        cv2.destroyAllWindows()
        self.cap.release()

if __name__ == "__main__":
    detector = PixelTracking()
    detector.run()
