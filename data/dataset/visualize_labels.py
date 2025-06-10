#!/usr/bin/env python3.7
"""
Visualize labels for model training

Usage:
python3.7  visualize_labels.py -f train/*jpg
press 'q' to exit
press 't' to play/pause
press any other key to advance
"""
import argparse
import os

import cv2
import numpy as np

COLORS = [(0, 255, 0), (0, 0, 255), (0, 255, 255), (255, 255, 0)]


def boxes(label_file, image_height, image_width):
    if not os.path.exists(label_file):
        yield [
            0,
            0,
            image_width,
            image_height,
        ], 0 if "todo" in label_file else 1
        return

    with open(label_file) as f:
        lines = f.readlines()
    for line in lines:
        [label_class, center_x, center_y, width, height] = map(
            float, line.split(" ")
        )
        box = [
            int((center_x - width / 2.0) * image_width),
            int((center_y - height / 2.0) * image_height),
            int(width * image_width),
            int(height * image_height),
        ]
        yield box, label_class


def visualize_labels(image_file, label_file):
    frame = cv2.imread(image_file)
    height, width = frame.shape[0:2]

    for box, label_class in boxes(label_file, height, width):
        color = COLORS[int(label_class) % len(COLORS)]
        label = str(label_class)

        cv2.rectangle(frame, box, color, 2)
        cv2.putText(
            frame,
            label,
            (box[0], box[1] - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            color,
            2,
        )

    return frame


def main():
    # handle command line arguments
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "-f",
        "--files",
        nargs="+",
        required=False,
        help="path to input image/video",
    )
    ap.add_argument(
        "-d",
        "--directory",
        required=False,
        help="path to directory containing images and labels",
    )
    ap.add_argument(
        "-s",
        "--silent",
        action="store_true",
        help="Dont show anything",
    )
    args = ap.parse_args()

    pause = 0
    delay_ms = 3

    H, W = 400, 600
    blank = np.zeros((H, W, 3), dtype=np.uint8)
    video_out = cv2.VideoWriter(
        "output.mp4",
        cv2.VideoWriter_fourcc(*"mp4v"),
        30,
        (W, H),
    )

    if args.directory:
        image_files = [os.path.join(args.directory, f) for f in os.listdir(args.directory) if f.endswith(".jpg")]
        label_files = [f.replace(".jpg", ".txt") for f in image_files]
    elif args.files:
        image_files = args.files
        label_files = [f.replace(".jpg", ".txt") for f in image_files]
    else:
        raise ValueError("No input files or directory provided")

    for image_file, label_file in zip(image_files, label_files):
        try:
            visu = visualize_labels(image_file, label_file)
        except:
            continue

        # resize all image to same target while keeping aspect ratios
        h, w = visu.shape[0:2]
        ratio = min(H / float(h), W / float(w))
        visu = cv2.resize(visu, (0, 0), fx=ratio, fy=ratio)
        canvas = blank.copy()
        nh, nw = visu.shape[0:2]
        x = int((W - nw) / 2)
        y = int((H - nh) / 2)
        canvas[y : y + nh, x : x + nw] = visu

        # video_out.write(canvas)

        if args.silent:
            continue

        cv2.imshow("Labels", canvas)
        k = cv2.waitKey(pause)
        if k == ord("q"):
            video_out.release()
            return
        elif k == ord("t"):
            pause = delay_ms - pause

    video_out.release()


if __name__ == "__main__":
    main()
