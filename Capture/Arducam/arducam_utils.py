import threading

import ArducamSDK
from utils import *
import argparse
import time

class ArducamCamera:
    """
    A class to interface with Arducam cameras.

    Attributes:
    - isOpened: A boolean indicating whether the camera is opened or not.
    - running_: A boolean indicating whether the camera is running or not.
    - thread_: A thread object for image capture.
    - lock_: A lock to synchronize access to the buffer.
    - buffer_: A list to store captured image data and configurations.
    """

    def __init__(self):
        self.isOpened = False
        self.running_ = False
        self.thread_ = None
        self.lock_ = threading.Lock()
        self.buffer_ = []

    def openCamera(self, fname, index=0):
        """
        Open the camera.

        Args:
        - fname (str): Path to the camera configuration file.
        - index (int): Index of the camera to open (default: 0).

        Returns:
        bool: True if the camera is opened successfully, False otherwise.
        """
        self.isOpened, self.handle, self.cameraCfg, self.color_mode = camera_initFromFile(fname, index)
        return self.isOpened

    def start(self):
        """
        Start capturing images from the camera.
        """
        if not self.isOpened:
            raise RuntimeError("The camera has not been opened.")

        self.running_ = True
        ArducamSDK.Py_ArduCam_setMode(self.handle, ArducamSDK.CONTINUOUS_MODE)

        # Start the capture thread
        self.thread_ = threading.Thread(target=self.capture_thread)
        self.thread_.start()


    def read(self, timeout=1500):
        """
        Read a captured image from the buffer.

        Args:
        - timeout (int): Maximum time (in milliseconds) to wait for an image.

        Returns:
        tuple: A tuple containing (success, data, cfg), where success is a boolean indicating whether the read was successful,
        data is the image data, and cfg is the image configuration.
        """
        if not self.running_:
            raise RuntimeError("The camera is not running.")

        start_time = time.time()
        while True:
            with self.lock_:
                if len(self.buffer_) > 0:
                    # Get the earliest image from the buffer
                    data, cfg = self.buffer_.pop()


                    return (True, data, cfg)

            if time.time() - start_time > timeout / 1000.0:
                return (False, None, None)

    def stop(self):
        """
        Stop capturing images from the camera.
        """
        if not self.running_:
            raise RuntimeError("The camera is not running.")

        self.running_ = False

        # Wait for the capture thread to finish
        self.thread_.join()

    def closeCamera(self):
        """
        Close the camera.
        """
        if not self.isOpened:
            raise RuntimeError("The camera has not been opened.")

        if (self.running_):
            self.stop()

        self.isOpened = False
        ArducamSDK.Py_ArduCam_close(self.handle)
        self.handle = None

    def capture_thread(self):
        """
        Thread function to capture images continuously.
        """
        ret = ArducamSDK.Py_ArduCam_beginCaptureImage(self.handle)

        if ret != 0:
            self.running_ = False
            raise RuntimeError("Error beginning capture, Error : {}".format(GetErrorString(ret)))

        print("Capture began, Error : {}".format(GetErrorString(ret)))

        while self.running_:
            ret = ArducamSDK.Py_ArduCam_captureImage(self.handle)
            if ret > 255:
                print("Error capture image, Error : {}".format(GetErrorString(ret)))
                if ret == ArducamSDK.USB_CAMERA_USB_TASK_ERROR:
                    break
            elif ret > 0:
                # Get the captured image data and configuration
                ret, data, cfg = ArducamSDK.Py_ArduCam_readImage(self.handle)
                ArducamSDK.Py_ArduCam_del(self.handle)
                size = cfg['u32Size']
                if ret != 0 or size == 0:
                    continue

                with self.lock_:
                    # Add the image data and configuration to the buffer
                    self.buffer_.append((data, cfg))


    def setCtrl(self, func_name, val):
            """
            Set camera control parameters.

            Args:
            - func_name (str): Name of the control function.
            - val: Value to set.

            Returns:
            bool: True if successful, False otherwise.
            """
            return ArducamSDK.Py_ArduCam_setCtrl(self.handle, func_name, val)

    def dumpDeviceInfo(self):
        """
        Print device information.
        """
        USB_CPLD_I2C_ADDRESS=0x46
        cpld_info={}
        ret, version = ArducamSDK.Py_ArduCam_readReg_8_8(
            self.handle, USB_CPLD_I2C_ADDRESS, 0x00)
        ret, year = ArducamSDK.Py_ArduCam_readReg_8_8(
            self.handle, USB_CPLD_I2C_ADDRESS, 0x05)
        ret, mouth = ArducamSDK.Py_ArduCam_readReg_8_8(
            self.handle, USB_CPLD_I2C_ADDRESS, 0x06)
        ret, day = ArducamSDK.Py_ArduCam_readReg_8_8(
            self.handle, USB_CPLD_I2C_ADDRESS, 0x07)

        cpld_info["version"] = "v{}.{}".format(version>>4, version & 0x0F)
        cpld_info["year"] = year
        cpld_info["mouth"] = mouth
        cpld_info["day"] = day

        print(cpld_info)

        ret, data = ArducamSDK.Py_ArduCam_getboardConfig(
            self.handle, 0x80, 0x00, 0x00, 2
        )

        usb_info={}
        usb_info["fw_version"] = "v{}.{}".format((data[0] & 0xFF), (data[1] & 0xFF))
        usb_info["interface"] = 2 if self.cameraCfg["usbType"] == 4 else 3
        usb_info["device"] = 3 if self.cameraCfg["usbType"] == 3 or self.cameraCfg["usbType"] == 4 else 2

        print(usb_info)

    def getCamInformation(self):
        """
        Get camera information.
        """
        self.version = ArducamSDK.Py_ArduCam_readReg_8_8(self.handle, 0x46, 00)[1]
        self.year = ArducamSDK.Py_ArduCam_readReg_8_8(self.handle, 0x46, 5)[1]
        self.mouth = ArducamSDK.Py_ArduCam_readReg_8_8(self.handle, 0x46, 6)[1]
        self.day = ArducamSDK.Py_ArduCam_readReg_8_8(self.handle, 0x46, 7)[1]
        cpldVersion = "V{:d}.{:d}\t20{:0>2d}/{:0>2d}/{:0>2d}".format(self.version >> 4, self.version & 0x0F, self.year,
                                                                     self.mouth, self.day)
        return cpldVersion

    def getMipiDataInfo(self):
        """
        Get MIPI data information.
        """
        mipiData = {"mipiDataID": "",
                    "mipiDataRow": "",
                    "mipiDataCol": "",
                    "mipiDataClk": "",
                    "mipiWordCount": "",
                    "mFramerateValue": ""}
        self.getCamInformation()
        cpld_version = self.version & 0xF0
        date = (self.year * 1000 + self.mouth * 100 + self.day)
        if cpld_version not in [0x20, 0x30]:
            return None
        if cpld_version == 0x20 and date < (19 * 1000 + 7 * 100 + 8):
            return None
        elif cpld_version == 0x30 and date < (19 * 1000 + 3 * 100 + 22):
            return None

        mipiDataID = ArducamSDK.Py_ArduCam_readReg_8_8(self.handle, 0x46, 0x1E)[1]
        mipiData["mipiDataID"] = hex(mipiDataID)

        rowMSB = ArducamSDK.Py_ArduCam_readReg_8_8(self.handle, 0x46, 0x21)[1]
        rowLSB = ArducamSDK.Py_ArduCam_readReg_8_8(self.handle, 0x46, 0x22)[1]
        mipiDataRow = ((rowMSB & 0xFF) << 8) | (rowLSB & 0xFF)
        mipiData["mipiDataRow"] = str(mipiDataRow)

        colMSB = ArducamSDK.Py_ArduCam_readReg_8_8(self.handle, 0x46, 0x1F)[1]
        colLSB = ArducamSDK.Py_ArduCam_readReg_8_8(self.handle, 0x46, 0x20)[1]
        mipiDataCol = ((colMSB & 0xFF) << 8) | (colLSB & 0xFF)
        mipiData["mipiDataCol"] = str(mipiDataCol)

        # after 2020/06/22
        if cpld_version == 0x20 and date < (20 * 1000 + 6 * 100 + 22):
            return mipiData
        elif cpld_version == 0x30 and date < (20 * 1000 + 6 * 100 + 22):
            return mipiData

        mipiDataClk = ArducamSDK.Py_ArduCam_readReg_8_8(self.handle, 0x46, 0x27)[1]
        mipiData["mipiDataClk"] = str(mipiDataClk)

        if (cpld_version == 0x30 and date >= (21 * 1000 + 3 * 100 + 1)) or (
                cpld_version == 0x20 and date >= (21 * 1000 + 9 * 100 + 6)):
            wordCountMSB = ArducamSDK.Py_ArduCam_readReg_8_8(self.handle, 0x46, 0x25)[1]
            wordCountLSB = ArducamSDK.Py_ArduCam_readReg_8_8(self.handle, 0x46, 0x26)[1]
            mipiWordCount = ((wordCountMSB & 0xFF) << 8) | (wordCountLSB & 0xFF)
            mipiData["mipiWordCount"] = str(mipiWordCount)

        if date >= (21 * 1000 + 6 * 100 + 22):
            fpsMSB = ArducamSDK.Py_ArduCam_readReg_8_8(self.handle, 0x46, 0x2A)[1]
            fpsLSB = ArducamSDK.Py_ArduCam_readReg_8_8(self.handle, 0x46, 0x2B)[1]
            fps = (fpsMSB << 8 | fpsLSB) / 4.0
            fpsResult = "{:.1f}".format(fps)
            mipiData["mFramerateValue"] = fpsResult
        return mipiData
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Arducam Camera Interface")
    parser.add_argument("--config_file", type=str, required=True, help="Path to the camera configuration file")
    args = parser.parse_args()

    camera = ArducamCamera()
    camera.openCamera(args.config_file)
    camera.start()

    success, data, cfg = camera.read()
    if success:
        print("Image captured successfully.")

    camera.stop()
    camera.closeCamera()

