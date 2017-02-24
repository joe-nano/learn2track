import argparse
import textwrap
import os
import numpy as np
from os.path import join as pjoin

import nibabel as nib

from dipy.viz import actor
from dipy.viz import window

interactive_session = False
rng = np.random.RandomState(42)
subsample = 500

def buildArgsParser():
    description = textwrap.dedent("""
        Build an article figure
    """)

    p = argparse.ArgumentParser(description=description)
    p.add_argument('anat', help='anatomy file used as background (T1/FA.nii.gz).')
    p.add_argument('path', help='folder containing the segmented bundles')
    p.add_argument('--output', default='fig2.png', help='output image. Default: %(default)s')
    return p


if __name__ == "__main__":
    parser = buildArgsParser()
    args = parser.parse_args()

    #anat = "/home/marc/research/dat/neuroimaging/ismrm2015_challenge/T1.nii.gz"
    #path = "/home/marc/research/dat/neuroimaging/ismrm2015_challenge/bundles"
    anat = args.anat
    path = args.path
    files = [pjoin(path, f) for f in os.listdir(path)]
    # Keep only "left" bundle
    files = [f for f in files if "right" not in f and "_VB_" in f]
    print("{} files detected".format(len(files)))


    bundle2color = {}
    bundle2color["UF"] = (174, 174, 69)
    bundle2color["OR"] = (255, 127, 0)
    bundle2color["Cingulum"] = (38, 129, 85)
    bundle2color["POPT"] = (227, 26, 28)
    bundle2color["CA"] = (166, 206, 227)
    bundle2color["ILF"] = (111, 4, 22)
    bundle2color["SCP"] = (51, 160, 44)
    bundle2color["FPT"] = (31, 120, 180)
    bundle2color["CC"] = (253, 191, 111)
    bundle2color["CP"] = (255, 127, 0)
    bundle2color["SLF"] = (116, 203, 203)
    bundle2color["ICP"] = (106, 61, 154)
    bundle2color["CST"] = (251, 154, 153)
    bundle2color["Fornix"] = (106, 61, 154)

    # Utils functions
    def add_actors(ren, bundles):
        for name in bundles:
            streamlines = None
            for filename in files:
                if name in os.path.basename(filename):
                    print("Loading: {}".format(os.path.basename(filename)))
                    streamlines = nib.streamlines.load(filename).streamlines
                    break
            if streamlines is None:
                print("Bundle not found: {}".format(name))
                continue

            idx = np.arange(len(streamlines))
            rng.shuffle(idx)
            streamlines = streamlines[idx][:subsample]
            color = bundle2color[name]
            act = actor.streamtube(streamlines, colors=np.asarray(color)/255., opacity=1., linewidth=0.3,)
            ren.add(act)


    def take_snapshot(bundles, interact_with=False):
        if interact_with:
            showm = window.ShowManager(ren, size=resolution, reset_camera=False)
            showm.start()
            ren.camera_info()

        snapshot_fname = "_".join(bundles) + ".png"
        print("Saving {}".format(snapshot_fname))
        window.snapshot(ren, fname=snapshot_fname, size=resolution, offscreen=True, order_transparent=False)
        return snapshot_fname


    # Prepare VTK scene
    resolution = (800, 600)
    ren = window.Renderer()

    # Load anatomy
    anat = nib.load(anat)
    slicer = actor.slicer(anat.get_data(), affine=anat.affine)

    ########################################
    # Generate all subfigures sequentially #
    ########################################
    snapshots = []

    ###
    # Make subfigure (a) UF, ILF, SLF
    ###
    bundles = ["UF", "ILF", "SLF"]
    ren.clear()
    add_actors(ren, bundles)

    # Select slice to display.
    slicer.display(x=anat.shape[0]-109)
    ren.add(slicer)

    # Place camera.
    ren.set_camera(position=(-366.69, -110.28, 67.20),
                   focal_point=(-106.26, -111.35, 75.79),
                   view_up=(-0.03, -0.00, 1.00))

    fname = take_snapshot(bundles, interact_with=interactive_session)
    snapshots.append(fname)

    ###
    # Make subfigure (b) OR, SCP, ICP
    ###
    bundles = ["OR", "SCP", "ICP"]
    ren.clear()
    add_actors(ren, bundles)

    # Select slice to display.
    slicer.display(x=anat.shape[0]-89)
    ren.add(slicer)

    # Place camera.
    ren.set_camera(position=(-366.69, -110.28, 67.20),
                   focal_point=(-106.26, -111.35, 75.79),
                   view_up=(-0.03, -0.00, 1.00))

    fname = take_snapshot(bundles, interact_with=interactive_session)
    snapshots.append(fname)

    ###
    # Make subfigure (c) Cingulum
    ###
    bundles = ["Cingulum"]
    ren.clear()
    add_actors(ren, bundles)

    # Select slice to display.
    slicer.display(x=anat.shape[0]-89)
    ren.add(slicer)

    # Place camera.
    ren.set_camera(position=(-366.69, -110.28, 67.20),
                   focal_point=(-106.26, -111.35, 75.79),
                   view_up=(-0.03, -0.00, 1.00))

    fname = take_snapshot(bundles, interact_with=interactive_session)
    snapshots.append(fname)

    # Make subfigure (d) FPT, CST, POPT (sagittal)
    # Make subfigure (e) FPT, CST, POPT (coronal)
    # Make subfigure (f) CC
    # Make subfigure (g) Fornix, CA, CP


    # Merge all snapshots
    from PIL import Image

    images = map(Image.open, snapshots)
    widths, heights = zip(*(i.size for i in images))

    total_width = sum(widths)
    max_height = max(heights)

    new_im = Image.new('RGB', (total_width, max_height))

    x_offset = 0
    for im in images:
      new_im.paste(im, (x_offset,0))
      x_offset += im.size[0]

    new_im.save(args.output)