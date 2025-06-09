from picamera2 import Picamera2

picam2 = Picamera2()
video_config = picam2.create_video_configuration(
    main={"size": (1280, 720)},
    controls={"FrameRate": 30.0}
)
picam2.configure(video_config)
picam2.start_preview(True)
picam2.start()
input("Press Enter to stop...")
picam2.stop()
