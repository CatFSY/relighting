
## ⚙️ Installation
```bash
git clone --recursive https://github.com/CatFSY/relighting.git

# This step is same as 2DGS/3DGS
# Please be aware that the submodules/diff-surfel-rasterization is slightly different from the original version in 2DGS.
conda env create --file environment.yml
conda activate irgs

# Install diff-surfel-rasterization and simple-knn
pip install submodules/diff-surfel-rasterization submodules/simple-knn

# Install raytracing (for Ref-Gaussian in stage 1)
pip install submodules/raytracing

# Install 2D Gaussian Ray Tracer
cd submodules/surfel_tracer && rm -rf ./build && mkdir build && cd build && cmake .. && make && cd ../ && cd ../../
pip install submodules/surfel_tracer
```



`relight_single.py` accepts one trained IRGS model and one EXR/HDR environment.
The camera may remain fixed or interpolate through the saved dataset poses.
Environment rotation can be enabled independently, including at the same time
as pose interpolation.

```bash
# Fixed camera, rotating environment
CUDA_VISIBLE_DEVICES=0 python relight_single.py \
  -m outputs/SCENE/irgs \
  --envmap assets/env_map/envmap3.exr \
  --output outputs/SCENE/irgs/relight_fixed_envrotate.mp4 \
  --pose-mode fixed --env-rotate

# Interpolated camera poses and rotating environment at the same time
CUDA_VISIBLE_DEVICES=0 python relight_single.py \
  -m outputs/SCENE/irgs \
  --envmap assets/env_map/envmap3.exr \
  --output outputs/SCENE/irgs/relight_pose_envrotate.mp4 \
  --pose-mode interpolate --steps-per-pair 6 --env-rotate
```

Omit `--env-rotate` to keep the environment fixed. `--pose-mode interpolate`
changes the camera pose; the object remains fixed in its trained coordinate
system.

### Assembled trajectory scene

`relight_assembled.py` is the public entry for SO101 link/object/table assembly.
It defaults to a full trajectory and the all-rigid IAS layout. The default
environment is the `fill0.35/key2500` EXR. Environment motion can be added to
the object trajectory.

```bash
# Full trajectory, fixed environment
CUDA_VISIBLE_DEVICES=0 python relight_assembled.py \
  --trajectory-set guiji2 \
  --out outputs/guiji2_relight

# Full trajectory and continuously rotating environment
CUDA_VISIBLE_DEVICES=0 python relight_assembled.py \
  --trajectory-set guiji2 \
  --trajectory-env-deg-per-sec 90 \
  --out outputs/guiji2_relight_envrotate
```

## 📜 BibTeX
```bibtex
@inproceedings{gu2024IRGS,
  title={IRGS: Inter-Reflective Gaussian Splatting with 2D Gaussian Ray Tracing},
  author={Gu, Chun and Wei, Xiaofei and Zeng, Zixuan and Yao, Yuxuan and Zhang, Li},
  booktitle={CVPR},
  year={2025},
}
```
# relighting
