from transformers import BlipProcessor, BlipForConditionalGeneration
import streamlit as st
from PIL import Image

model_name = "Salesforce/blip-image-captioning-base"


#title
st.title("Image Caption Generator")

#subtitle
st.markdown("This application helps you to generate caption for your image.")

@st.cache_data(show_spinner="Loading the app..")
def load_processor(model_name): 
    processor = BlipProcessor.from_pretrained(model_name) 
    return processor

@st.cache_data(show_spinner="App Loading..")  
def load_model(model_name): 
    model = BlipForConditionalGeneration.from_pretrained(model_name) 
    return model

processor = load_processor(model_name)
model = load_model(model_name)

#image uploader
image = st.file_uploader(label = "Upload your image here",type=['png','jpg','jpeg'])

if image is not None:

    input_image = Image.open(image) 
    # st.image(input_image) #display image

    with st.spinner("Creating the captions... "):

        raw_image = input_image.convert('RGB')
        inputs = processor(raw_image, return_tensors="pt")

        out = model.generate(**inputs)
        result = processor.decode(out[0], skip_special_tokens=True)

        st.write(result)
        
   
else:
    st.write("Upload an Image first")


st.caption("Made by Gourav Chouhan ")





    




# if __name__ == "__main__":
#     model_name = "Salesforce/blip-image-captioning-base"
#     load_model(model_name)
  


