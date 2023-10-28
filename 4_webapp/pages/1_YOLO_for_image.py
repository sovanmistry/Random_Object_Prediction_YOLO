import streamlit as st
from yolo_predictions import YOLO_Pred
# necessary to import image
from PIL import Image
# For converting image to Array
import numpy as np


st.set_page_config(page_title="YOLO Object Detection",
                   layout='wide',
                   page_icon='./images/object.png')

st.header('Get Object Detection for any Image')
st.write('Please Upload Image to get detections')

with st.spinner('Please wait while your model is loading'):
    yolo = YOLO_Pred(onnx_model='./models/best.onnx',
                    data_yaml='./models/data.yaml')
    st.balloons()



def upload_image():
    # Upload Image
    image_file = st.file_uploader(label = "Upload Image")
    if image_file is not None:
        size_mb = image_file.size
        file_details = {"File Name" :image_file.name,
                        "File Type" :image_file.type,
                        "File Size" :"{:,.2f} MB".format(size_mb)}
        
        # st.json(file_details)

        # Validate Image File
        if file_details['File Type'] in ('image/jpeg' or 'image/png'):
            st.success(" Valid Image File Type (png, jpeg or jpg) ")
            return {"file":image_file,
                    "details":file_details}
        
        else:
            st.error("Inavlid Image File Type")
            st.error("Please Upload it in png, jpeg or jpg")
            return None
        

# Main Function
def main(): 
    object = upload_image()

    if object:
        prediction = False
        image_obj = Image.open(object['file'])
        # st.image(image_obj)

        col1, col2 = st.columns(2)

        with col1: 
            st.info("Preview of Image")
            st.image(image_obj)

        with col2: 
            st.subheader("Check Below for File Details")
            st.json(object['details'])
            button = st.button("Get Detection from Yolo")
            
            if button: 
                # Below command convert image into Array
                image_array = np.array(image_obj)
                pred_img = yolo.predictions(image_array)
                # Below command convert Array into Object
                pred_img_obj = Image.fromarray(pred_img)
                prediction = True

        if prediction:
            st.subheader("Predicted Image")
            st.caption("your Predicted Image from YOLO V5")
            st.image(pred_img_obj)      


# Without Below Main wont work
if __name__ == "__main__":
    main()
