import os
import numpy as np
import imageio


def load_tat_data(basedir='/data/tanks_and_temples', testskip=8):


    def parse_intrinsics(filepath):  # , trgt_sidelength, invert_y=False):
        # Get camera intrinsics
        with open(filepath, 'r') as file:
            full_intrinsics = np.array(list(map(float, file.readline().split())))
        full_intrinsics = full_intrinsics.reshape((4, 4))
        print(full_intrinsics)
        return full_intrinsics

        #     f, cx, cy = list(map(float, file.readline().split()))[:3]
        #     grid_barycenter = np.array(list(map(float, file.readline().split())))
        #     near_plane = float(file.readline())
        #     scale = float(file.readline())
        #     height, width = map(float, file.readline().split())

        #     try:
        #         world2cam_poses = int(file.readline())
        #     except ValueError:
        #         world2cam_poses = None

        # if world2cam_poses is None:
        #     world2cam_poses = False

        # world2cam_poses = bool(world2cam_poses)

        # print(cx,cy,f,height,width)

        # cx = cx / width * trgt_sidelength
        # cy = cy / height * trgt_sidelength
        # f = trgt_sidelength / height * f

        # fx = f
        # if invert_y:
        #     fy = -f
        # else:
        #     fy = f

        # Build the intrinsic matrices
        # full_intrinsic = np.array([[fx, 0., cx, 0.],
        #                            [0., fy, cy, 0],
        #                            [0., 0, 1, 0],
        #                            [0, 0, 0, 1]])

        # return full_intrinsic, grid_barycenter, scale, near_plane, world2cam_poses


    def load_pose(filename):
        assert os.path.isfile(filename)
        nums = open(filename).read().split()
        return np.array([float(x) for x in nums]).reshape([4,4]).astype(np.float32)


    # H = 512
    # W = 512
    tat_base = '{}/train/'.format(basedir)

    # full_intrinsic, grid_barycenter, scale, near_plane, world2cam_poses = \
    #     parse_intrinsics(os.path.join(tat_base, 'intrinsics', '000001.txt'), H)
    # print(full_intrinsic, grid_barycenter, scale, near_plane, world2cam_poses)
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
    i_split = [np.arange(counts[i], counts[i+1]) for i in range(len(all_imgs))]

    imgs = np.concatenate(all_imgs, 0)
    poses = np.concatenate([poses, testposes, testposes], 0)
    render_poses = testposes

    print(poses.shape, imgs.shape)

    return imgs, poses, render_poses, [H, W, focal], i_split


