# laika
A repository for all the code related to robot dog: Laika


# stream mac webcam to autovideosink

gst-launch-1.0 avfvideosrc ! videoconvert ! video/x-raw,format=BGR,width=640,height=480,framerate=15/1 ! autovideosink

# receive video from udp source (BGR)

gst-launch-1.0 -v udpsrc port=5004 caps = "application/x-rtp, media=(string)video, clock-rate=(int)90000, encoding-name=(string)H264, payload=(int)96" ! rtph264depay ! decodebin ! videoconvert ! autovideosink

# receive video from udp source (RGB)

gst-launch-1.0 -v udpsrc port=5004 caps="application/x-rtp, media=(string)video, clock-rate=(int)90000, encoding-name=(string)H264, payload=(int)96" ! rtph264depay ! decodebin ! videoconvert ! video/x-raw, format=(string)RGB ! autovideosink


# stream mac webcam to appsink

gst-launch-1.0 -e nvarguscamerasrc ! 'video/x-raw(memory:NVMM), width=(int)640, height=(int)640, format=(string)NV12, framerate=12/1' ! nvvidconv ! 'video/x-raw(memory:NVMM), format=NV12' ! nvv4l2h264enc insert-sps-pps=true ! h264parse ! rtph264pay pt=96 ! udpsink host=192.168.3.2 port=5004 auto-multicast=true sync=false
