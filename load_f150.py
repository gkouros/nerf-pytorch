import os
import numpy as np
import imageio


def load_f150_data(basedir='/data/f150', testskip=8):


    def parse_intrinsics(filepath):  # , trgt_sidelength, invert_y=False):
        # Get camera intrinsics
        with open(filepath, 'r') as file:
            full_intrinsics = np.array(list(map(float, file.readline().split())))
        full_intrinsics = full_intrinsics.reshape((4, 4))
        print(full_intrinsics)
        return full_intrinsics

    def load_pose(filename):
        assert os.path.isfile(filename)
        nums = open(filename).read().split()
        return np.array([float(x) for x in nums]).reshape([4,4]).astype(np.float32)

    tat_base = '{}/train/'.format(basedir)
    full_intrinsics = parse_intrinsics(
        os.path.join(tat_base, 'intrinsics', '000001.txt'))#, H)
    focal = full_intrinsics[0, 0]
    # print(H, W, focal)

    def dir2poses(posedir):
        poses = np.stack([load_pose(os.path.join(posedir, f)) for f in sorted(
            os.listdir(posedir)) if f.endswith('txt')], 0)
        transf = np.array([
            [1, 0, 0, 0],
            [0, -1, 0, 0],
            [0, 0, -1, 0],
            [0, 0, 0, 1.],
        ])
        poses = poses @ transf
        poses = poses[:, :3, :4].astype(np.float32)
        return poses

    posedir = os.path.join(tat_base, 'pose')
    poses = dir2poses(posedir)
    testposes = dir2poses('{}/test/pose'.format(basedir))
    testposes = testposes[::testskip]

    imgfiles = [f for f in sorted(os.listdir(os.path.join(tat_base, 'rgb')))
                if f.endswith('png')]
    imgs = np.stack([imageio.imread(os.path.join(tat_base, 'rgb', f))/255.
                     for f in imgfiles], 0).astype(np.float32)
    H, W = imgs[0].shape[:2]

    testimgd = '{}/test/rgb'.format(basedir)
    imgfiles = [f for f in sorted(os.listdir(testimgd)) if f.endswith('png')]
    testimgs = np.stack([imageio.imread(os.path.join(testimgd, f))/255.
                         for f in imgfiles[::testskip]], 0).astype(np.float32)

    all_imgs = [imgs, testimgs, testimgs]
    counts = [0] + [x.shape[0] for x in all_imgs]
    counts = np.cumsum(counts)
    print(counts)
    i_split = [np.arange(counts[i], counts[i+1]) for i in range(len(all_imgs))]

    imgs = np.concatenate(all_imgs, 0)
    poses = np.concatenate([poses, testposes, testposes], 0)
    render_poses = testposes

    print(poses.shape, imgs.shape)

    return imgs, poses, render_poses, [H, W, focal], i_split


