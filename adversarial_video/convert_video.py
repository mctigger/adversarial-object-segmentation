from os.path import join

import av


target_dir = './data/video/video2014'


def video_to_frames(video_path):
    container = av.open(video_path)
    for i, frame in enumerate(container.decode(video=0)):
        if i > 500:
            break

        frame_path = join(target_dir, 'frame-{:09d}.jpg'.format(frame.index))
        img = frame.to_image().resize((1920 // 2, 1080 // 2))

        print("Saving to {}".format(frame_path))
        img.save(frame_path)


video_to_frames('./data/video/0002-20170519-2.mp4')