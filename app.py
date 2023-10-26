import cv2
import numpy as np
import streamlit as st
from skimage.metrics import structural_similarity as ssim
from PIL import Image


def resize_image(image, width, height):
    return cv2.resize(image, (width, height), interpolation=cv2.INTER_AREA)


def compare_images(image1, image2):
    # Convert the images to grayscale
    gray1 = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(image2, cv2.COLOR_BGR2GRAY)

    # Compute the Structural Similarity Index (SSIM) between the two images
    (score, _) = ssim(gray1, gray2, full=True)
    return score


def main():
    st.title("Signature Verification")

    # Allow user to upload two image files
    i = st.camera_input("Take a picture", key="camera1")
    i_2 = st.camera_input("Take another picture", key="camera2")
    # gen = st.file_uploader("Upload Genuine Signature", type=["png", "jpg", "jpeg"])
    # forged = st.file_uploader("Upload Forged Signature", type=["png", "jpg", "jpeg"])

    if i_2 and i:
        genuine_signature = cv2.imdecode(np.fromstring(i_2.read(), np.uint8), cv2.IMREAD_COLOR)
        forged_signature = cv2.imdecode(np.fromstring(i.read(), np.uint8), cv2.IMREAD_COLOR)

        # Display the uploaded images in Streamlit
        st.image(Image.open(i_2), caption="Signature 1", use_column_width=True)
        st.image(Image.open(i), caption="Signature 2", use_column_width=True)

        # Resize both images to a common size for comparison
        common_width = 200
        common_height = 100
        genuine_signature_resized = resize_image(genuine_signature, common_width, common_height)
        forged_signature_resized = resize_image(forged_signature, common_width, common_height)

        # Compare the resized images using SSIM
        similarity_score = compare_images(genuine_signature_resized, forged_signature_resized)

        # Set a threshold to classify real and fake signatures
        threshold = 0.55

        # Display the result
        if similarity_score > threshold:
            st.write("The signature is genuine.")
            st.write(similarity_score)
        else:
            st.write("The signature is fake.")
            st.write(similarity_score)


if __name__ == "__main__":
    main()
