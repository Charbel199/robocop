
import os
from pathlib import Path
import shutil


def move_txt(s):

    src_path =f'/home/charbel199/projs/robocop/ml/yolov5/dataset/annotated_rover2/images/{s}'
    trg_path =f'/home/charbel199/projs/robocop/ml/yolov5/dataset/annotated_rover2/labels/{s}'


    for src_file in Path(src_path).glob('*.txt'):
        head = Path(src_file).name

        t = os.path.join(trg_path, head)

        shutil.move(src_file, t)

move_txt('train')
move_txt('test')
move_txt('val')