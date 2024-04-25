import cv2 as cv
import subprocess

class FindDevices:
    def test_device(self, source):
        """
        Tests if a video device is accessible.

        Args:
            source (str): Video device source.

        Returns:
            str: Source if device is accessible, None otherwise.
        """
        try:
            cap = cv.VideoCapture(source)
            if cap is not None and cap.isOpened():
                return source
        except Exception as e:
            pass

    def get_available_devices(self):
        """
        Retrieves a list of available video devices.

        Returns:
            list: List of available video devices.
        """
        output = subprocess.check_output("ls /dev/video*", shell=True)
        video_devices = output.decode().split()
        return video_devices

    def test_all_devices(self):
        """
        Tests all available video devices.

        Returns:
            list: List of accessible video device sources.
        """
        video_devices = self.get_available_devices()
        return [self.test_device(dev) for dev in video_devices]

if __name__ == "__main__":
    tester = FindDevices()
    accessible_devices = tester.test_all_devices()
    print(accessible_devices)
