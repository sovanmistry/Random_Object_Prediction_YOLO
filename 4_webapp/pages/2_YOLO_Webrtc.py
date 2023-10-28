from streamlit_webrtc import webrtc_streamer
import av
from yolo_predictions import YOLO_Pred

#Load Yolo model
yolo = YOLO_Pred(onnx_model='./models/best.onnx',
                 data_yaml='./models/data.yaml')


def video_frame_callback(frame):
    # Converted image to array
    img = frame.to_ndarray(format="bgr24")
    
    # Operations
    # flipped = img[::-1,:,:]
    pred_img = yolo.predictions(img)

    return av.VideoFrame.from_ndarray(pred_img, format="bgr24")


webrtc_streamer(key="example", video_frame_callback=video_frame_callback)