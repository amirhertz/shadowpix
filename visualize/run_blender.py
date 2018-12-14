import os
import sys
from functools import partial
from multiprocessing.dummy import Pool
from subprocess import call
import glob
import argparse


project_root = '/home/amir/projects/shadowpix'
parser = argparse.ArgumentParser()
parser.add_argument('--blender_path', type=str, default='/opt/blender/blender', help='')
parser.add_argument('--script_path', type=str, default=project_root + '/visualize/render.py', help='')
parser.add_argument('--blend_path', type=str, default=project_root + '/visualize/stage.blend', help='')
parser.add_argument('--input_folder', type=str, default=project_root + '/models/test.obj', help='')
parser.add_argument('--output_folder', type=str, default=project_root + '/renders', help='')
opt = parser.parse_args()


def render(input_obj_dir, output_folder):
    print('Generating rendering commands...')
    if os.path.isfile(input_obj_dir):
        fullfile = input_obj_dir
    else:
        return
    if not os.path.isdir(output_folder):
        os.makedirs(output_folder)
    file_id, file_extension = os.path.splitext(os.path.basename(fullfile))
    output_file = os.path.join(output_folder, file_id + '.png')
    command = ['%s %s --background --python %s -- %s %s > /dev/null 2>&1' % (opt.blender_path, opt.blend_path, opt.script_path, fullfile, output_file)]
    print(command[0])
    print('Rendering')
    pool = Pool(1)
    for idx, return_code in enumerate(pool.imap(partial(call, shell=True), command)):
        if return_code != 0:
            print('Rendering command (\"%s\") failed' % (command[idx]))
    print('done!')


if __name__ == "__main__":
    render(opt.input_folder, opt.output_folder)