import cv2
import numpy as np
import mediapipe as mp
import streamlit as st
from scipy.interpolate import splprep, splev
from streamlit_image_comparison import image_comparison
from io import BytesIO
from PIL import Image

# --- Mediapipe setup ---
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
    static_image_mode=False,
    max_num_faces=1,
    refine_landmarks=False,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

LIP_LANDMARKS = [61,146,91,181,84,17,314,405,321,375,291,409,270,269,267,0,37,39,40,185,
                 78,95,88,178,87,14,317,402,318,324,308]

def rgb_to_bgr(rgb_color):
    return (rgb_color[2], rgb_color[1], rgb_color[0])

def apply_realistic_lipstick(image, lip_color=(180,60,60), alpha=0.6, blur_intensity=5, results=None):
    if results is None or not results.multi_face_landmarks:
        return image
    h, w, _ = image.shape
    mask = np.zeros((h, w), dtype=np.float32)
    for face_landmarks in results.multi_face_landmarks:
        lip_points = [(int(face_landmarks.landmark[idx].x*w),
                       int(face_landmarks.landmark[idx].y*h)) for idx in LIP_LANDMARKS]
        if len(lip_points) >= 3:
            lip_points = np.array(lip_points)
            tck,_ = splprep(lip_points.T, s=1.0)
            x_new, y_new = splev(np.linspace(0,1,200), tck)
            smooth_lips = np.vstack((x_new,y_new)).T.astype(np.int32)
            cv2.fillPoly(mask, [smooth_lips], 1.0)
    mask = cv2.GaussianBlur(mask, (0,0), blur_intensity)
    mask = np.clip(mask, 0, 1)
    ys, xs = np.where(mask>0)
    output = image.copy().astype(np.float32)
    for c in range(3):
        output[:,:,c] = (1 - mask * alpha) * output[:,:,c] + mask * alpha * lip_color[c]
    gloss = np.zeros_like(output)
    if len(ys) > 0:
        gloss_center = (int(np.mean(xs)), int(np.mean(ys)))
        cv2.ellipse(gloss, gloss_center, (15,5), 0, 0, 360, (255,255,255), -1)
        gloss = cv2.GaussianBlur(gloss, (0,0), 7)
        output = cv2.addWeighted(output, 1, gloss, 0.2*alpha, 0)
    return np.clip(output,0,255).astype(np.uint8)

# --- 70+ realistic lipstick shades ---
natural_colors = {
    "Soft Red": (180, 60, 60), "Rose": (180, 90, 120), "Coral": (200, 120, 90), "Nude": (190, 170, 150),
    "Peach": (220, 150, 130), "Berry": (150, 50, 80), "Mauve": (160, 100, 130), "Brick Red": (140, 50, 50),
    "Cherry Red": (200, 30, 50), "Wine": (100, 20, 40), "Plum": (120, 40, 70), "Fuchsia": (220, 60, 120),
    "Magenta": (200, 40, 100), "Hot Pink": (255, 50, 180), "Bubblegum": (255, 100, 180), "Candy Pink": (255, 140, 180),
    "Rosewood": (135, 60, 75), "Rust": (180, 70, 50), "Terracotta": (170, 90, 70), "Brick Orange": (190, 90, 60),
    "Coral Pink": (240, 130, 110), "Salmon": (250, 140, 120), "Apricot": (255, 170, 120), "Peach Nude": (245, 180, 150),
    "Beige": (220, 190, 160), "Caramel": (200, 160, 120), "Mocha": (150, 110, 90), "Coffee": (120, 80, 60),
    "Chestnut": (140, 90, 70), "Mahogany": (100, 50, 40), "Cranberry": (170, 40, 70), "Mulberry": (130, 20, 50),
    "Dusty Rose": (190, 120, 130), "Mauve Pink": (200, 130, 140), "Lavender Pink": (220, 140, 160),
    "Deep Rose": (150, 50, 90), "Dark Berry": (100, 30, 60), "Brick Mauve": (160, 80, 90), "Warm Red": (210, 50, 60),
    "Classic Red": (220, 20, 30), "Tomato Red": (230, 60, 50), "Strawberry": (255, 50, 70), "Coral Red": (240, 90, 80),
    "Raspberry": (180, 30, 70), "Peony": (255, 110, 150), "Blush Pink": (255, 140, 160), "Orchid": (210, 120, 200),
    "Lavender": (200, 150, 220), "Plum Berry": (120, 30, 80), "Mulberry Mist": (140, 20, 70), "Cranberry Glow": (180, 30, 60),
    "Sangria": (150, 0, 50), "Carmine": (160, 0, 50), "Red Velvet": (200, 20, 40), "Hot Coral": (255, 80, 60),
    "Peach Glow": (255, 180, 130), "Apricot Blush": (255, 170, 130), "Nude Beige": (230, 200, 180), "Soft Mauve": (180, 130, 140),
    "Berry Crush": (150, 30, 70), "Rose Petal": (210, 100, 120), "Cocoa": (120, 70, 60), "Chestnut Brown": (130, 80, 70)
}
color_bgr_dict = {k: rgb_to_bgr(v) for k,v in natural_colors.items()}

# --- Presets ---
presets = {
    "Romantic Look": {"color": color_bgr_dict["Rose"], "opacity": 0.6},
    "Bold Red": {"color": color_bgr_dict["Classic Red"], "opacity": 0.75},
    "Casual Nude": {"color": color_bgr_dict["Nude"], "opacity": 0.5},
    "Party Pink": {"color": color_bgr_dict["Hot Pink"], "opacity": 0.7}
}

# --- Streamlit UI ---
st.title("💄 Virtual Lipstick Try-On (70+ Shades + Presets)")
st.write("Select a color or preset, upload your photo, or try webcam!")

# --- Preset Selector ---
selected_preset = st.selectbox("✨ Choose a Makeup Preset", ["None"] + list(presets.keys()))

selected_color = None
alpha_lip = 0.6

if selected_preset != "None":
    selected_color = presets[selected_preset]["color"]
    alpha_lip = presets[selected_preset]["opacity"]

# --- Color Palette ---
palette_container = st.container()
with palette_container:
    color_names = list(natural_colors.keys())
    rows = (len(color_names) // 8) + 1
    for r in range(rows):
        cols = st.columns(8)
        for c in range(8):
            idx = r*8 + c
            if idx >= len(color_names):
                break
            name = color_names[idx]
            rgb = natural_colors[name]
            r_val, g_val, b_val = rgb

            with cols[c]:
                st.markdown(
                    f"""
                    <div style="background-color: rgb({r_val},{g_val},{b_val});
                                width: 40px; height: 25px;
                                border-radius: 5px; border:1px solid #000;
                                margin-bottom:4px;"></div>
                    """,
                    unsafe_allow_html=True
                )
                if st.button(name, key=name):
                    selected_color = color_bgr_dict[name]

if selected_color is None:
    selected_color = color_bgr_dict["Soft Red"]

alpha_lip = st.slider("Lipstick Opacity", 0.4, 0.8, alpha_lip)

# --- Image Upload ---
uploaded_file = st.file_uploader("Upload your photo", type=["jpg","jpeg","png"])
if uploaded_file is not None:
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, 1)
    results = face_mesh.process(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    output = apply_realistic_lipstick(img, selected_color, alpha=alpha_lip, blur_intensity=5, results=results)

    # Show Before/After comparison
    image_comparison(
        img1=cv2.cvtColor(img, cv2.COLOR_BGR2RGB),
        img2=cv2.cvtColor(output, cv2.COLOR_BGR2RGB),
        label1="Before",
        label2="After"
    )

    # Save option
    pil_image = Image.fromarray(cv2.cvtColor(output, cv2.COLOR_BGR2RGB))
    buf = BytesIO()
    pil_image.save(buf, format="PNG")
    byte_im = buf.getvalue()
    st.download_button("💾 Download Lipstick Image", data=byte_im, file_name="lipstick_tryon.png", mime="image/png")

# --- Webcam Live Lipstick ---
use_webcam = st.checkbox("📷 Use Webcam (Live Lipstick)")
if use_webcam:
    stframe = st.empty()
    cap = cv2.VideoCapture(0)
    frame_count = 0
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frame = cv2.flip(frame,1)
            if frame_count % 3 == 0:
                results = face_mesh.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            makeup_frame = apply_realistic_lipstick(frame, selected_color, alpha=alpha_lip, blur_intensity=5, results=results)
            stframe.image(cv2.cvtColor(makeup_frame, cv2.COLOR_BGR2RGB), channels="RGB", use_container_width=True, output_format="JPEG")
            frame_count += 1
    finally:
        cap.release()

