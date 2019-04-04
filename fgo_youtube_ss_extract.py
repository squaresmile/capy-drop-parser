import os
import argparse
import cv2
import numpy as np
from tqdm import tqdm
import youtube_dl

YOUTUBE_DL_OPTIONS = {
    "format": "bestvideo[ext=mp4][height<=1080]/best[ext=mp4][height<=1080]",
    "outtmpl": "%(uploader)s-%(id)s.%(ext)s",
}
THRESHOLD = 0.2


def recognize_drop_text(frame, template):
    res = cv2.matchTemplate(frame, template, cv2.TM_SQDIFF_NORMED)
    loc = np.where(res < THRESHOLD)
    # loc is empty if the frame doesn't match
    return loc[0].size > 0


def extract_drop_screen(file, output_folder):
    cap = cv2.VideoCapture(file)
    template = cv2.imread("drop_text.png")
    total_frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    with tqdm(total=total_frame_count) as pbar:
        count = 1
        drop_scene = []
        while cap.isOpened():
            ret, frame = cap.read()
            if ret:
                detected = recognize_drop_text(frame, template)
                if detected:
                    # There will be 20-30 frames with the drop text in the same scene
                    drop_scene.append({"count": count, "frame": frame})
                    count += 1
                if drop_scene and not detected:
                    chosen = drop_scene[int(len(drop_scene) * 0.7)]
                    cv2.imwrite(os.path.join(output_folder, f"{file}_{chosen['count']}.png"), chosen['frame'])
                    drop_scene = []
                pbar.update(1)
            else:
                break


    cap.release()
    cv2.destroyAllWindows()


def run(link, quest):
    quest_folder = os.path.join("input", str(quest))
    if not os.path.isdir(quest_folder):
        os.mkdir(quest_folder)
    if os.path.exists(link):
        file_name = link
        extract_drop_screen(file_name, quest_folder)
    else:
        try:
            ydl = youtube_dl.YoutubeDL(YOUTUBE_DL_OPTIONS)
            ydl.add_default_info_extractors()
            info = ydl.extract_info(link)
            file_name = f"{info['uploader']}-{info['id']}.mp4"
            extract_drop_screen(file_name, quest_folder)
        except (youtube_dl.utils.UnsupportedError, youtube_dl.utils.DownloadError):
            pass


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Get drop screenshots from Youtube link")
    parser.add_argument("-l", "--link", help="File or Youtube link to download")
    parser.add_argument("-q", "--quest", help="Output quest folder in input folder", default="video_screenshot")
    args = parser.parse_args()

    run(args.link, args.quest)
