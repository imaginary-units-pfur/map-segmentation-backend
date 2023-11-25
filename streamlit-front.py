import streamlit as st
import numpy as np
import cv2
import time
import PIL

PAGE_TITLE = "imaginary_units@PFUR/map-segmentation"

import modelst as model

def overlay(image, mask, color, alpha, resize=None):
    """Combines image and its segmentation mask into a single image.
    https://www.kaggle.com/code/purplejester/showing-samples-with-segmentation-mask-overlay

    Params:
        image: Training image. np.ndarray,
        mask: Segmentation mask. np.ndarray,
        color: Color for segmentation mask rendering.  tuple[int, int, int] = (255, 0, 0)
        alpha: Segmentation mask's transparency. float = 0.5,
        resize: If provided, both image and its mask are resized before blending them together.
        tuple[int, int] = (1024, 1024))

    Returns:
        image_combined: The combined image. np.ndarray

    """
    color = color[::-1]
    colored_mask = np.expand_dims(mask, 0).repeat(3, axis=0)
    colored_mask = np.moveaxis(colored_mask, 0, -1)
    masked = np.ma.MaskedArray(image, mask=colored_mask, fill_value=color)
    image_overlay = masked.filled()

    if resize is not None:
        image = cv2.resize(image.transpose(1, 2, 0), resize)
        image_overlay = cv2.resize(image_overlay.transpose(1, 2, 0), resize)

    image_combined = cv2.addWeighted(image, 1 - alpha, image_overlay, alpha, 0)

    return image_combined

def main():
    print("hit", time.ctime())
    st.set_page_config(page_title=PAGE_TITLE, layout="wide")
    st.title(PAGE_TITLE)
    st.link_button("View on GitHub", "https://github.com/imaginary-units-pfur/map-segmentation")
    if 'segments' not in st.session_state:
        st.session_state.segments = None
    uploaded_file = st.file_uploader("Choose a file")
    if uploaded_file is not None:
        # To read file as bytes:
        bytes_data = uploaded_file.getvalue()
        src_image = model.img_from_bytes(bytes_data, None)[0]
        if st.button('Run Segmentation'):
            print("Running segmentation: ", time.time())
            st.session_state.segments = None
            start_time_1 = time.time()
            st.write("Started at: ", start_time_1)
            instance_image = model.segment(bytes_data)
            end_time_1 = time.time() - start_time_1
            st.session_state.segments = instance_image
            st.write("Time taken:", end_time_1)
            print("Time taken: ", end_time_1)

        if st.session_state.segments is not None:
            instance_image = st.session_state.segments

            overlaid = overlay(src_image, instance_image, (255,0,0), 0.5)
            st.image(overlaid, caption="Overlay", width=1024, clamp=True)
            png = cv2.imencode(".png", overlaid)[1].tobytes()
            st.download_button("Download overlay as PNG", png, 'overlay.png', 'image/png')

            st.image(instance_image, width=128, caption="Raw segmentation")
            png = cv2.imencode(".png", instance_image)[1].tobytes()
            st.download_button("Download mask as PNG", png, 'mask.png', 'image/png')




if __name__ == "__main__":
    main()