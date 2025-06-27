import cv2
import glob
import numpy as np
from moviepy import VideoFileClip, AudioFileClip
from moviepy.video.io.ImageSequenceClip import ImageSequenceClip
import tempfile
import os

class SpeakerStatus:
    def __init__ (self, length):
        self.speaking_status = [0] * length

    def __str__(self):
        return(f"{self.speaking_status}")

    def get_status(self):
        return self.speaking_status
    
    def update(self, segment):
        start, end, speaker = segment

        for i in range(start, end):
            if i < len(self.speaking_status):
                if self.speaking_status[i] != 0:
                    if self.speaking_status[i] == speaker:
                        print("This should not happen")
                    else:
                        self.speaking_status[i] = 3 # both speakers 
                else:
                    self.speaking_status[i] = speaker 
            else:
                print(f"Error: index {i} out of bounds")

def timestamp_to_frame(timestamps, fps):
    data = []
    for start, end, speaker in timestamps:
        start_idx = int(start * fps)
        end_idx = int(end * fps)
        data.append((start_idx, end_idx, speaker))
    return data

def read_timestamps(file_path):
    data = []
    with open(file_path, "r") as file:
        for line in file:
            # Example line: "1.03 to 1.53 : SPEAKER_01"
            if "to" in line and ":" in line:
                time_part, speaker = line.strip().split(" : ")
                start, end = time_part.split(" to ")

                if speaker == "SPEAKER_00":
                    speaker = 1
                elif speaker == "SPEAKER_01":
                    speaker = 2
                else:
                    print(f"Error: invalid speaker {speaker}")
                    exit(1)

                data.append((float(start), float(end), speaker))
    return data

def get_video_info(video_path):
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        print("Error: Could not open video.")
    else:
        # Get total number of frames
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        # Get frames per second (fps)
        fps = cap.get(cv2.CAP_PROP_FPS)

        # print(f"Total Frames: {total_frames}")
        # print(f"FPS: {fps}")

        cap.release()
    
    return total_frames, fps

def annotate_video_by_speaker_with_audio(input_path, output_path, labels, font_scale=2, thickness=3):
    """
    Annotate each frame with speaker labels and preserve original audio using updated moviepy API.
    
    Args:
        input_path (str): Path to the input mp4 video.
        output_path (str): Path to save the final annotated video.
        labels (list[int]): A list of integers (0–3), one per video frame.
        font_scale (int): Font size for the text overlay.
        thickness (int): Thickness of the annotation text.
    """
    # Open video using OpenCV
    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        raise IOError(f"Cannot open video file: {input_path}")

    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps         = cap.get(cv2.CAP_PROP_FPS)
    width       = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height      = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    if len(labels) != frame_count:
        raise ValueError(f"Label count ({len(labels)}) does not match video frame count ({frame_count})")

    font = cv2.FONT_HERSHEY_SIMPLEX
    label_map = {
        0: "No One Speaking",
        1: "Person 1 Speaking",
        2: "Person 2 Speaking",
        3: "Both Speaking"
    }

    frames = []
    idx = 0

    print("Annotating frames...")
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        label = labels[idx]
        text = label_map.get(label, "Unknown")

        # Get text size and position
        (text_width, text_height), _ = cv2.getTextSize(text, font, font_scale, thickness)
        x = (width - text_width) // 2
        y = (height + text_height) // 2

        # Annotate frame
        cv2.putText(frame, text, (x, y), font, font_scale, (255, 255, 255), thickness, cv2.LINE_AA)

        # Convert BGR (OpenCV) to RGB (moviepy expects RGB)
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frames.append(frame_rgb)

        idx += 1

    cap.release()

    # Create video clip from frames
    print("Creating video clip...")
    video_clip = ImageSequenceClip(frames, fps=fps)

    # Extract audio using moviepy
    print("Extracting audio...")
    original_clip = VideoFileClip(input_path)
    video_clip = video_clip.with_audio(original_clip.audio)

    # Write final video
    print("Writing final video with audio...")
    video_clip.write_videofile(output_path, codec="libx264", audio_codec="aac")

    print(f"Annotated video with audio saved to: {output_path}")

def main():
    # video_path  = "C:\\Users\\odink\\Documents\\Adobe\\Premiere Pro\\25.0\\035_client_w_server_audio.mp4"
    
    # total_frames, fps = get_video_info(video_path)
    # timestamps = read_timestamps("./outputs/010_segments.txt")
    # frames_indices = timestamp_to_frame(timestamps, fps)
    # # print(frames_indices)
    # speaking_status = SpeakerStatus(total_frames)
    
    # for segment in frames_indices:
    #     speaking_status.update(segment)

    # status = speaking_status.get_status()
    
    # annotate_video_by_speaker_with_audio(video_path, "./annotated_video/test_1.mp4", status)
    
    folder_path = 'C:\\Users\\odink\\Documents\\Adobe\\Premiere Pro\\25.0'
    mp4_files = glob.glob(os.path.join(folder_path, '*.mp4'))

    for video_path in mp4_files:
        file_name_wo_ext = os.path.splitext(os.path.basename(video_path))[0]
        if os.path.exists(f"./segmentations/{file_name_wo_ext}.txt") and not os.path.exists(f"./annotated_video/{file_name_wo_ext}_annotated.mp4"):
            total_frames, fps = get_video_info(video_path)
            timestamps = read_timestamps(f"./segmentations/{file_name_wo_ext}.txt")
            frames_indices = timestamp_to_frame(timestamps, fps)
            speaking_status = SpeakerStatus(total_frames)
            for segment in frames_indices:
                speaking_status.update(segment)

            status = speaking_status.get_status()

            annotate_video_by_speaker_with_audio(video_path, f"./annotated_video/{file_name_wo_ext}_annotated.mp4", status)

if __name__ == "__main__":
    main()
