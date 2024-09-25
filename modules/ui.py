import os
import customtkinter as ctk
from typing import Callable, Tuple
import cv2
from PIL import Image, ImageOps

import modules.globals
import modules.metadata
from modules.face_analyser import get_one_face
from modules.processors.frame.core import get_frame_processors_modules
from modules.utilities import is_image, resolve_relative_path, has_image_extension


ROOT = None
ROOT_HEIGHT = 700
ROOT_WIDTH = 600

PREVIEW = None
PREVIEW_MAX_HEIGHT = 700
PREVIEW_MAX_WIDTH = 1200
PREVIEW_DEFAULT_WIDTH = 720
PREVIEW_DEFAULT_HEIGHT = 1000

RECENT_DIRECTORY_SOURCE = None
RECENT_DIRECTORY_TARGET = None
RECENT_DIRECTORY_OUTPUT = None

preview_label1 = None
preview_label2 = None
source_label = None
status_label = None

img_ft, vid_ft = modules.globals.file_types


def init(start: Callable[[], None], destroy: Callable[[], None]) -> ctk.CTk:
    global ROOT, PREVIEW

    ROOT = create_root(start, destroy)
    PREVIEW = create_preview(ROOT)

    return ROOT


def create_root(start: Callable[[], None], destroy: Callable[[], None]) -> ctk.CTk:
    global source_label, target_label, status_label

    ctk.deactivate_automatic_dpi_awareness()
    ctk.set_appearance_mode('system')
    ctk.set_default_color_theme(resolve_relative_path('ui.json'))

    root = ctk.CTk()
    root.minsize(ROOT_WIDTH, ROOT_HEIGHT)
    root.title("AI换脸演示")
    root.configure()
    root.protocol('WM_DELETE_WINDOW', lambda: destroy())

    source_label = ctk.CTkLabel(root,text=None)
    source_label.place(relx=0.3, rely=0.1, relwidth=0.4, relheight=0.3)

    select_face_button = ctk.CTkButton(root, text='请选择伪造人物', cursor='hand2', command=lambda: select_source_path())
    select_face_button.place(relx=0.3, rely=0.4, relwidth=0.4, relheight=0.1)

    enhancer_value = ctk.BooleanVar(value=modules.globals.fp_ui['face_enhancer'])
    enhancer_switch = ctk.CTkSwitch(root, text='面部增强(不要提前打开)', variable=enhancer_value, cursor='hand2',
                                    command=lambda: update_tumbler('face_enhancer', enhancer_value.get()))
    enhancer_switch.place(relx=0.3, rely=0.65)

    live_button = ctk.CTkButton(root, text='开始换脸', cursor='hand2', command=lambda: webcam_preview())
    live_button.place(relx=0.40, rely=0.8, relwidth=0.2, relheight=0.05)

    return root


def create_preview(parent: ctk.CTkToplevel) -> ctk.CTkToplevel:
    global preview_label1, preview_slider, preview_label2

    preview = ctk.CTkToplevel(parent)
    preview.withdraw()
    preview.title('实时AI换脸')
    preview.configure()
    preview.protocol('WM_DELETE_WINDOW', lambda: toggle_preview())
    preview.resizable(width=True, height=True)

    # 创建左框架
    left_frame = ctk.CTkFrame(preview)
    left_frame.pack(side='left', fill='y', expand=True)

    # 创建右框架
    right_frame = ctk.CTkFrame(preview)
    right_frame.pack(side='right', fill='y', expand=True)

    preview_label1 = ctk.CTkLabel(right_frame, text=None)
    preview_label1.pack(fill='y', expand=True)

    preview_label2 = ctk.CTkLabel(left_frame, text=None)
    preview_label2.pack(fill='y', expand=True)


    return preview


def update_status(text: str) -> None:
    status_label.configure(text=text)
    ROOT.update()


def update_tumbler(var: str, value: bool) -> None:
    modules.globals.fp_ui[var] = value


def select_source_path() -> None:
    global RECENT_DIRECTORY_SOURCE, img_ft, vid_ft

    PREVIEW.withdraw()
    source_path = ctk.filedialog.askopenfilename(title='select an source image', initialdir=RECENT_DIRECTORY_SOURCE,
                                                   filetypes=[img_ft])

    if is_image(source_path):
        modules.globals.source_path = source_path
        RECENT_DIRECTORY_SOURCE = os.path.dirname(modules.globals.source_path)
        image = render_image_preview(modules.globals.source_path, (200, 200))
        source_label.configure(image=image)
    else:
        modules.globals.source_path = None
        source_label.configure(image=None)

def check_and_ignore_nsfw(target, destroy: Callable = None) -> bool:
    ''' Check if the target is NSFW.
    TODO: Consider to make blur the target.
    '''
    from numpy import ndarray
    from modules.predicter import predict_image, predict_video, predict_frame
    if type(target) is str:  # image/video file path
        check_nsfw = predict_image if has_image_extension(target) else predict_video
    elif type(target) is ndarray:  # frame object
        check_nsfw = predict_frame
    if check_nsfw and check_nsfw(target):
        if destroy: destroy(to_quit=False)  # Do not need to destroy the window frame if the target is NSFW
        update_status('Processing ignored!')
        return True
    else:
        return False


def fit_image_to_size(image, width: int, height: int):
    if width is None and height is None:
        return image
    h, w, _ = image.shape
    ratio_h = 0.0
    ratio_w = 0.0
    if width > height:
        ratio_h = height / h
    else:
        ratio_w = width / w
    ratio = max(ratio_w, ratio_h)
    new_size = (int(ratio * w), int(ratio * h))
    return cv2.resize(image, dsize=new_size)


def render_image_preview(image_path: str, size: Tuple[int, int]) -> ctk.CTkImage:
    image = Image.open(image_path)
    if size:
        image = ImageOps.fit(image, size, Image.LANCZOS)
    return ctk.CTkImage(image, size=image.size)


def render_video_preview(video_path: str, size: Tuple[int, int], frame_number: int = 0) -> ctk.CTkImage:
    capture = cv2.VideoCapture(video_path)
    if frame_number:
        capture.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
    has_frame, frame = capture.read()
    if has_frame:
        image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        if size:
            image = ImageOps.fit(image, size, Image.LANCZOS)
        return ctk.CTkImage(image, size=image.size)
    capture.release()
    cv2.destroyAllWindows()


def toggle_preview() -> None:
    if PREVIEW.state() == 'normal':
        PREVIEW.withdraw()
    elif modules.globals.source_path and modules.globals.target_path:
        init_preview()



def init_preview() -> None:
    if is_image(modules.globals.target_path):
        preview_slider.pack_forget()


def webcam_preview():
    if modules.globals.source_path is None:
        # No image selected
        return

    global preview_label1, PREVIEW, preview_label2

    camera = cv2.VideoCapture(0)  # Use index for the webcam (adjust the index accordingly if necessary)
    camera.set(cv2.CAP_PROP_FRAME_WIDTH, PREVIEW_DEFAULT_WIDTH)  # Set the width of the resolution
    camera.set(cv2.CAP_PROP_FRAME_HEIGHT, PREVIEW_DEFAULT_HEIGHT)  # Set the height of the resolution
    camera.set(cv2.CAP_PROP_FPS, 60)  # Set the frame rate of the webcam

    preview_label1.configure(width=PREVIEW_DEFAULT_WIDTH,
                             height=PREVIEW_DEFAULT_HEIGHT)  # Reset the preview image before startup
    preview_label2.configure(width=PREVIEW_DEFAULT_WIDTH,
                             height=PREVIEW_DEFAULT_HEIGHT)

    PREVIEW.deiconify()  # Open preview window

    frame_processors = get_frame_processors_modules(modules.globals.frame_processors)

    source_image = None  # Initialize variable for the selected face image

    while camera:
        ret, frame = camera.read()
        if not ret:
            break

        # Select and save face image only once
        if source_image is None and modules.globals.source_path:
            source_image = get_one_face(cv2.imread(modules.globals.source_path))

        rect_frame = frame.copy()
        closest_face = get_one_face(frame)
        if closest_face is not None:
            top, right, bottom, left = int(closest_face.bbox[1]), int(closest_face.bbox[0]), int(
                closest_face.bbox[3]), int(closest_face.bbox[2])
            cv2.rectangle(rect_frame, (left, top), (right, bottom), (0, 0, 255), 2)
        else:
            pass

        if modules.globals.live_resizable:
            rect_frame = fit_image_to_size(rect_frame, PREVIEW.winfo_width(), PREVIEW.winfo_height())

        temp_frame = frame.copy()  # Create a copy of the frame

        if modules.globals.live_mirror:
            temp_frame = cv2.flip(temp_frame, 1)  # horizontal flipping

        if modules.globals.live_resizable:
            temp_frame = fit_image_to_size(temp_frame, PREVIEW.winfo_width(), PREVIEW.winfo_height())

        for frame_processor in frame_processors:
            temp_frame = frame_processor.process_frame(source_image, temp_frame)

        image1 = cv2.cvtColor(temp_frame, cv2.COLOR_BGR2RGB)
        # Convert the image to RGB format to display it with Tkinter
        image1 = Image.fromarray(image1)
        image1 = ImageOps.contain(image1, (temp_frame.shape[1], temp_frame.shape[0]), Image.LANCZOS)
        image1 = ctk.CTkImage(image1, size=image1.size)

        image2 = cv2.cvtColor(rect_frame, cv2.COLOR_BGR2RGB)
        image2 = Image.fromarray(image2)
        image2 = ImageOps.contain(image2, (rect_frame.shape[1], rect_frame.shape[0]), Image.LANCZOS)
        image2 = ctk.CTkImage(image2, size=image2.size)

        preview_label1.configure(image=image1)
        preview_label2.configure(image=image2)
        ROOT.update()

        if PREVIEW.state() == 'withdrawn':
            break

    camera.release()
    PREVIEW.withdraw()  # Close preview window when loop is finished
