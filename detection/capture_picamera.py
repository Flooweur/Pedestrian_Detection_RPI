import time
from PIL import Image
from picamera2 import Picamera2


with Picamera2() as camera:
    camera.start()
    time.sleep(1)

    image = camera.capture_array("main")
    im = Image.fromarray(image)
    timestamp = time.strftime("%Y%m%d-%H%M%S")

    im.save(f"images/captured_image_{timestamp}.png")
