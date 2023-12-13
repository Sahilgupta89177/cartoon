import cv2
import numpy as np
import streamlit as st
from collections import defaultdict
from scipy import stats
import base64

# Functions for cartoonization

def update_c(C, hist):
    while True:
        groups = defaultdict(list)

        for i in range(len(hist)):
            if hist[i] == 0:
                continue
            d = np.abs(C - i)
            index = np.argmin(d)
            groups[index].append(i)

        new_C = np.array(C)
        for i, indice in groups.items():
            if np.sum(hist[indice]) == 0:
                continue
            new_C[i] = int(np.sum(indice * hist[indice]) / np.sum(hist[indice]))

        if np.sum(new_C - C) == 0:
            break
        C = new_C

    return C, groups

def K_histogram(hist):
    alpha = 0.001
    N = 80
    C = np.array([128])

    while True:
        C, groups = update_c(C, hist)

        new_C = set()
        for i, indice in groups.items():
            if len(indice) < N:
                new_C.add(C[i])
                continue

            z, pval = stats.normaltest(hist[indice])
            if pval < alpha:
                left = 0 if i == 0 else C[i - 1]
                right = len(hist) - 1 if i == len(C) - 1 else C[i + 1]
                delta = right - left
                if delta >= 3:
                    c1 = (C[i] + left) / 2
                    c2 = (C[i] + right) / 2
                    new_C.add(c1)
                    new_C.add(c2)
                else:
                    new_C.add(C[i])
            else:
                new_C.add(C[i])
        if len(new_C) == len(C):
            break
        else:
            C = np.array(sorted(new_C))
    return C

def caart(img, bilateral_filter_value, canny_threshold1, canny_threshold2, erode_kernel_size, bw_filter):
    kernel = np.ones((2, 2), np.uint8)
    output = np.array(img)
    x, y, c = output.shape

    for i in range(c):
        output[:, :, i] = cv2.bilateralFilter(output[:, :, i], bilateral_filter_value, 150, 150)

    edge = cv2.Canny(output, canny_threshold1, canny_threshold2)
    output = cv2.cvtColor(output, cv2.COLOR_RGB2HSV)

    hists = []

    hist, _ = np.histogram(output[:, :, 0], bins=np.arange(180 + 1))
    hists.append(hist)
    hist, _ = np.histogram(output[:, :, 1], bins=np.arange(256 + 1))
    hists.append(hist)
    hist, _ = np.histogram(output[:, :, 2], bins=np.arange(256 + 1))
    hists.append(hist)

    C = []
    for h in hists:
        C.append(K_histogram(h))

    output = output.reshape((-1, c))
    for i in range(c):
        channel = output[:, i]
        index = np.argmin(np.abs(channel[:, np.newaxis] - C[i]), axis=1)
        output[:, i] = C[i][index]
    output = output.reshape((x, y, c))
    if bw_filter:
        output = cv2.cvtColor(output[:, :, 0], cv2.COLOR_GRAY2RGB)
    else:
        output = cv2.cvtColor(output, cv2.COLOR_HSV2RGB)

    contours, _ = cv2.findContours(edge, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    cv2.drawContours(output, contours, -1, 0, thickness=1)

    for i in range(3):
        output[:, :, i] = cv2.erode(output[:, :, i], kernel, iterations=erode_kernel_size)

    return output

def main():
    st.title('Real-Time Video Cartoonizer')

    video_capture_object = cv2.VideoCapture(0)
    out = None  # VideoWriter object

    st.sidebar.title("Cartoonize Parameters")
    bilateral_filter_value = st.sidebar.slider("Bilateral Filter Value", 5, 50, 9)
    canny_threshold1 = st.sidebar.slider("Canny Threshold 1", 0, 255, 100)
    canny_threshold2 = st.sidebar.slider("Canny Threshold 2", 0, 255, 200)
    erode_kernel_size = st.sidebar.slider("Erode Kernel Size", 1, 5, 2)
    bw_filter = st.sidebar.checkbox("Black and White Filter", False)
    rotation_angle = st.sidebar.slider("Rotation Angle", -180, 180, 0)  # Add rotation slider

    start_stop_button = st.button("Start/Stop Video")

    stframe = st.empty()

    while start_stop_button:
        ret, frame = video_capture_object.read()
        if not ret:
            break

        # Apply rotation to the frame
        rows, cols, _ = frame.shape
        M = cv2.getRotationMatrix2D((cols / 2, rows / 2), rotation_angle, 1)
        frame = cv2.warpAffine(frame, M, (cols, rows))

        # Apply cartoonization to the rotated frame
        cartoon_frame = caart(frame, bilateral_filter_value, canny_threshold1, canny_threshold2, erode_kernel_size, bw_filter)

        if out is None:
            # Initialize VideoWriter if not done yet
            out = cv2.VideoWriter('cartoonizing-master/out.mp4', cv2.VideoWriter_fourcc(*'mp4v'), 30, (cartoon_frame.shape[1], cartoon_frame.shape[0]))

        out.write(cartoon_frame)

        # Display the rotated and cartoonized frame
        stframe.image(cartoon_frame, channels="BGR")

    # Release resources when the loop is stopped
    if out is not None:
        out.release()
    video_capture_object.release()

    # Add a download button for the recorded video
    st.markdown("## Download Recorded Video")
    download_button = st.button("Download Video")

    if download_button:
        st.markdown(get_binary_file_downloader_html('cartoonizing-master/out.mp4', 'Video'), unsafe_allow_html=True)

# Function to generate a download link for the recorded video
def get_binary_file_downloader_html(file_path, file_label='File'):
    with open(file_path, 'rb') as f:
        data = f.read()
    b64 = base64.b64encode(data).decode()
    href = f'<a href="data:file/mp4;base64,{b64}" download="{file_path}">Download {file_label}</a>'
    return href

if __name__ == '__main__':
    main()
