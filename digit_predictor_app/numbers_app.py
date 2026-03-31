import streamlit as st
import numpy as np
import joblib
from PIL import Image, ImageOps


st.set_page_config(page_title="Digit Predictor", page_icon="✍️")

st.title("Welcome to the Handwritten Digit Predictor! ✍️")
st.write("Upload a photo of ONE handwritten digit.")

# Tips visas för användare för bättre prediktion, eftersom modellen är tränad på tydliga bilder av siffror.
st.info("⚠️ Tips !! Use a clear, large digit on a white paper and crop the image so that the digit occupies most of the frame.")

# Nu laddars in den tränade modellen som jag sparade i Notebooken.
model = joblib.load("final_mnist_model.joblib")


# ==============================================================================
# Här processas den uppladdade bilden så att den matchar formatet som modellen tränades på (28x28 pixlar, gråskala, inverterade färger).

def preprocess_image(image):
    
    img = image.convert("L")  # "Grayscale" mode
    img = img.resize((28, 28))  # Resize till 28x28 (samma som MNIST)
    img = ImageOps.invert(img)  # Invertera färgerna så att det matchar MNIST datasetet (vit siffra på svart bakgrund)

    img_array = np.array(img)  # Konvertera till numpy array
    # Enkelt threshold för att göra bakgrunden mörkare och siffran tydligare
    img_array = np.where(img_array > 100, 255, 0)  # Detta görde mycket för att förbättra prediktionerna. Innan dess predikterades 8 för alla bilder.
    img_array = img_array.reshape(1, -1)  # Reshape till (1, 784) för att matcha modellens input
    
    processed_img = Image.fromarray(np.uint8(img_array.reshape(28, 28)))  # Konvertera tillbaka till bildformat

    return processed_img, img_array

# ==============================================================================

# ==============================================================================
# Här hanteras filuppladdningen och prediktionen. 
# Om en fil har laddats upp, så visas både originalbilden och den processade bilden (28x28) bredvid varandra.

# Den skapar "Drag and drp file here" rutan där användaren kan ladda upp en bild, och den accepterar endast png, jpg och jpeg format.
uploaded_file = st.file_uploader("Upload image here", type=["png", "jpg", "jpeg"])

if uploaded_file is not None:
    
    original_img = Image.open(uploaded_file)  # Öppna den uppladdade bilden
    
    processed_img, processed_array = preprocess_image(original_img)  # Processa bilden så att den matchar formatet som modellen tränades på (28x28 pixlar, gråskala, inverterade färger)

    col1, col2 = st.columns(2)  # Skapa två kolumner för att visa originalbilden och den processade bilden bredvid varandra så att jag förstår skillnaden.

    with col1:
        st.image(original_img, caption="Original image", width=200)
    with col2:
        st.image(processed_img, caption="Processed 28x28 image", width=200)

    if st.button("Predict digit"):  # Prediktion när användaren klickar på knappen
        prediction = model.predict(processed_array)[0]
        st.success(f"Predicted digit: {prediction}")
    
else:
    st.info("Please upload an image to start the prediction!")