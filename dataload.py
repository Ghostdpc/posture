from scipy.misc import imread, imresize, imsave
import numpy as np
import  os


def clipread(paths, offsets, size=(128, 171), crop_size=(112, 112), mode='RGB', interp='bilinear'):
    """
    data load
    :param paths: path to video frames
    :param offsets: Crop window offset in form of [from_H, to_H, from_W, to_W], example: (0, 112, 24, 136)
    :param size: size of out put
    :param crop_size:  size of the output cropped image
    :param mode: "rgb/L"
    :param interp: Interpolation to use for re-sizing, example: 'nearest', 'lanczos', 'bilinear', 'bicubic' or 'cubic'
    :return: Cropped clip (depth, crop_height, crop_width, channels) in float32 format, pixel values in [0, 255]
    """
    assert mode in ('RGB', 'L'), 'Mode is either RGB or L'

    clips = []

    x = 0
    y = 1
    for file_name in paths:
        # Read video frame

        try:
         im = imread(file_name, mode=mode)
         y+=1
        except OSError:
         im = imread(paths[0+x], mode=mode)
         x+=1      




        # Resize frame to init resolution and crop then resize to target resolution
        if mode == 'RGB':
            im = imresize(im, size=size, interp=interp)
            data = im[offsets[0]:offsets[1], offsets[2]:offsets[3], :]
            im = imresize(data, size=crop_size, interp=interp)
        else:
            im = imresize(im, size=size, interp=interp)
            data = im[offsets[0]:offsets[1], offsets[2]:offsets[3]]
            im = imresize(data, size=crop_size, interp=interp)

        clips.append(im)
    clips = np.array(clips, dtype=np.float32)
    if mode == 'RGB':
        return clips
    return np.expand_dims(clips, axis=3)


def randcrop(scales, size=(128, 171)):
    """
    Generate random offset for crop window
    :param scales: List of scales for crop window, example: (128, 112, 96, 84)
    :param size: Tuple, size of the image
    :return: Crop window offsets in form of (from_H, to_H, from_W, to_W), example: (0, 112, 24, 136)
    """
    scales = np.array(scales) if isinstance(scales, (list, tuple)) else np.array([scales])
    scale = scales[np.random.randint(len(scales))]
    height, width = size

    max_h = height - scale
    max_w = width - scale

    off_h = np.random.randint(max_h) if max_h > 0 else 0
    off_w = np.random.randint(max_w) if max_w > 0 else 0

    return off_h, off_h + scale, off_w, off_w + scale


def centercrop(scale, size=(128, 171)):
    """
    Generate center offset for crop window
    :param scale: Int, a scale for crop window, example: 112
    :param size: Tuple, size of the image, example: (128, 171)
    :return: Crop window offsets in form of (from_H, to_H, from_W, to_W), example: (8, 120, 29, 141)
    """
    height, width = size

    off_h = np.ceil((height - scale) / 2).astype(int)
    off_w = np.ceil((width - scale) / 2).astype(int)

    return off_h, off_h + scale, off_w, off_w + scale


def _demo_read_clip():
    """
   read video frames and return ararry of cropped frames
    """
    frame_dir = 'E:/proroot/dataset/traindata/hmdb/50_FIRST_DATES_sit_f_cm_np1_fr_med_24/'
    paths = [frame_dir +str (f + 1)+'.jpg' for f in range(0, 0 + 16)]

    # Random crop 16 times to see the differences
    clips = []
    for _ in range(16):
        offs = randcrop(scales=[128, 112, 96, 84], size=(128, 171))
        # offs = centercrop(scale=64, size=(128, 171))
        clip = clipread(paths=paths, offsets=offs, size=(128, 171), crop_size=(112, 112), mode='RGB')

        # Random flip left-right
        if np.random.rand(1, 1).squeeze() > 0.5:
            clip = np.flip(clip, axis=2)
        clips.append(np.hstack(clip))

    # Saving to a single images, each row is a clip
    clips = np.vstack(clips).squeeze()
    print('Min:', clips.min(), 'Max:', clips.max())
    imsave('E:/proroot/dataset/traindata/hmdb/50_FIRST_DATES_sit_f_cm_np1_fr_med_24/clips.png', clips)


if __name__ == '__main__':
    print('Demo video processing!')
    _demo_read_clip()
