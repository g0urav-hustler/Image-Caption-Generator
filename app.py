from transformers import BlipProcessor, BlipForConditionalGeneration
import streamlit as st
from PIL import Image

model_name = "Salesforce/blip-image-captioning-base"

# getting the model

#title
st.title("Image Caption Generator")

#subtitle
st.markdown("This application helps you to generate caption on your image")

#image uploader
image = st.file_uploader(label = "Upload your image here",type=['png','jpg','jpeg'])


@st.cache_data
def load_model(model_name): 
    processor = BlipProcessor.from_pretrained(model_name)
    model = BlipForConditionalGeneration.from_pretrained(model_name) 
    return [processor, model]

processor, model = load_model(model_name) #load model

if image is not None:

    input_image = Image.open(image) #read image
    # st.image(input_image) #display image

    with st.spinner("Creating the captions "):

        raw_image = input_image.convert('RGB')
        inputs = processor(raw_image, return_tensors="pt")

        out = model.generate(**inputs)
        result = processor.decode(out[0], skip_special_tokens=True)
        


        result_text = {"Caption1": result} #empty list for results


    

        st.write(result_text)
        
   
else:
    st.write("Upload an Image first")





    




# if __name__ == "__main__":
#     model_name = "Salesforce/blip-image-captioning-base"
#     load_model(model_name)
  


