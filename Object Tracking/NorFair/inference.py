import io, os
from base64 import b64encode
from IPython.display import HTMLos.system('wget "https://raw.githubusercontent.com/tryolabs/norfair/master/demos/colab/requirements.txt" -O requirements.txt')
os.system('pip install -r requirements.txt')

os.system('"https://drive.google.com/u/0/uc?id=1Jc5TAiwOZ-yUO6R_tG0zSW9Niv_HKTPV&export=download" -O sample.mp4')
os.system('ffmpeg -i sample.mp4 -ss 7 -t 10 sample_10s.mp4')

os.system('wget "https://raw.githubusercontent.com/tryolabs/norfair/master/demos/colab/demo.py"')
os.system('"https://raw.githubusercontent.com/tryolabs/norfair/master/demos/colab/draw.py"')
os.system('"https://raw.githubusercontent.com/tryolabs/norfair/master/demos/colab/yolo.py"')

os.system("python demo.py sample_10s.mp4 --classes 0")
os.system("ffmpeg -i ./sample_10s_out.mp4 -vcodec vp9 ./sample.webm")

with  io.open('sample.webm','r+b') as f:
    mp4 = f.read()
data_url = "data:video/webm;base64," + b64encode(mp4).decode()
HTML("""
<video width=800 controls>
      <source src="%s" type="video/webm">
</video>
""" % data_url)