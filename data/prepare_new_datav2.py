"""
Install:
pip install pandas
pip install fire
pip install opencv-python
pip install mediapipe
pip install tqdm

RUN:
python aslfrv4/data/prepare_new_data.py \
--data_dir /ivi/ilps/projects/ltl-mt/baohao/aslfr/ChicagoFSWild \
--output_path /ivi/ilps/projects/ltl-mt/baohao/aslfr/ChicagoFSWild/chicago.parquet.gzip
"""


import os
import fire
import numpy as np
import pandas as pd
import cv2
import mediapipe as mp
from tqdm import tqdm
from metadata import XY_POINT_LANDMARKS


mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_holistic = mp.solutions.holistic


def extract_keypoints(results):
    pose = np.array([[res.x, res.y, res.z] for res in results.pose_landmarks.landmark]).flatten() if results.pose_landmarks else np.zeros(33*3) * np.nan
    face = np.array([[res.x, res.y, res.z] for res in results.face_landmarks.landmark]).flatten() if results.face_landmarks else np.zeros(468*3) * np.nan
    lh = np.array([[res.x, res.y, res.z] for res in results.left_hand_landmarks.landmark]).flatten() if results.left_hand_landmarks else np.zeros(21*3) * np.nan
    rh = np.array([[res.x, res.y, res.z] for res in results.right_hand_landmarks.landmark]).flatten() if results.right_hand_landmarks else np.zeros(21*3) * np.nan
    return np.concatenate([face, pose, lh, rh])

def main(data_dir: str, output_path: str, start_idx: int, end_idx):
    df = pd.read_csv(os.path.join(data_dir, "batch.csv"))[start_idx:end_idx]

    with mp_holistic.Holistic(static_image_mode=True, model_complexity=2, refine_face_landmarks=False) as holistic:
        landmarks = []
        frames = []
        file_names = []
        for index, row in tqdm(df.iterrows()):
            image_files = []
            for i in range(row["number_of_frames"]):
                image_files.append(os.path.join(data_dir, row["filename"]) + "/" + str(i+1).zfill(4) + ".jpg")
                frames.append(i)
                file_names.append(row["filename"])
            for idx, file in enumerate(image_files):
                image = cv2.imread(file)
                image_height, image_width, _ = image.shape
                results = holistic.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
                landmarks.append(extract_keypoints(results))

    landmarks = np.vstack(landmarks)
    assert len(frames) == len(file_names) == np.shape(landmarks)[0]

    parquet_df = {}
    parquet_df["file_name"] = file_names
    parquet_df["frame"] = frames

    for i in range(478):
        for j, coord in enumerate(["x", "y", "z"]):
            parquet_df[coord + "_face_" + str(i)] = landmarks[:, i * 3 + j]
    for i in range(33):
        for j, coord in enumerate(["x", "y", "z"]):
            parquet_df[coord + "_pose_" + str(i)] = landmarks[:, 468 * 3 + i * 3 + j]
    for i in range(21):
        for j, coord in enumerate(["x", "y", "z"]):
            parquet_df[coord + "_left_hand_" + str(i)] = landmarks[:, 468 * 3 + 33 * 3 + i * 3 + j]
    for i in range(21):
        for j, coord in enumerate(["x", "y", "z"]):
            parquet_df[coord + "_right_hand_" + str(i)] = landmarks[:, 468 * 3 + 33 * 3 + 21 * 3 + i * 3 + j]

    parquet_df = pd.DataFrame(parquet_df)
    selected_columns = ["file_name"] + ["frame"] + XY_POINT_LANDMARKS
    filtered_parquet_df = parquet_df[selected_columns]
    filtered_parquet_df.to_parquet(output_path, compression='gzip')


if __name__ == "__main__":
    fire.Fire(main)