# Omni-AnyNet

Omni-AnyNet [1] is a network for omnidirectional stereo vision.
It extends the network AnyNet [2] by incorporating OmniGlasses, a set of carefully designed look up tables.
The repository was originally forked from: https://github.com/mileyan/AnyNet

The current version of OmniGlasses allows to process a stereo pair of images underlying the equiangular camera model.
Furthermore, it is currently required, that the cameras are arranged in a canonical stereo setup.
Hence, the cameras' x-axes (left-right) are collinear, and y-axes (top-down) are parallel.

If you use this work for an own publication, we kindly ask you to cite [1] and [2] (see Section References)

The following submodules are used by default:
`Omni_lut` / `OmniGlasses`: OmniGlasses of Omni-AnyNet
`map_processing`: Helper scripts for processing omnidirectional images and maps
(see .gitmodules for repository locations)

## Preparations

First, please download the sceneflow model from the [AnyNet webpage](https://github.com/mileyan/AnyNet) to
`checkpoint/sceneflow/sceneflow.tar`.
Then use the Bash script `prepare_machine.sh` that helps you to create an appropriate Conda environment and that compiles necessary modules. It was tested on Ubuntu 22.04.

For a test run (`run.sh`), you need to download a checkpoint from ~~here~~ (coming soon).
We trained and evaluated Omni-AnyNet on a converted dataset of THEOStereo [3]. This converted dataset can be downloaded from ~~here~~ (coming soon).
If you use this dataset in your work, please cite [1] and [3]. Thank you.


## Training / Inference / Evaluation

Scripts:

| Script name                   | Purpose                                            |
| ----------------------------- | -------------------------------------------------- |
| `create_dataset.sh`           | leftover from original AnyNet                      |
| `evaluate.sh`                 | evaluates a certain checkpoint / epoch             |
| `evaluate_current_version.sh` | helper script to select parameters for evaluate.sh |
| `prepare_machine.sh`          | generates Conda environment and compiles SPN       |
| `run.sh`                      | inference and time measurements                    |
| `start_finetune.sh`           | starts training / finetuning                       |


Most important environment variables (see each Bash script for default values):

| Variable               | Description                                                  | Used in                                  |
| ---------------------- | ------------------------------------------------------------ | ---------------------------------------- |
| `BASELINE`             | baseline / distance between cameras in stereo setup          | `start_finetune.sh`                      |
| `BETA`                 | loss function: Beta parameter of Smooth-L1-Loss              | `start_finetune.sh, evaluate.sh`         |
| `BSIZE`                | batch size (inference)                                       | `run.sh`                                 |
| `CHKPNT`               | path to initial checkpoint for training / finetuning         | `start_finetune.sh, evaluate.sh`         |
| `CONDA_ENV_NAME`       | name of Conda environment                                    | `run.sh, start_finetune.sh, evaluate.sh` |
| `CUDA_DEVICE_ORDER`    | oder of CUDA devices (usually `PCI_BUS_ID`)                  | `run.sh, start_finetune.sh, evaluate.sh` |
| `CUDA_VISIBLE_DEVICES` | indices of CUDA devices to use (e.g. 0,1 for GPU 0 and 1)    | `run.sh, start_finetune.sh, evaluate.sh` |
| `DATASET`              | path to the dataset for inference or training                | `run.sh, start_finetune.sh, evaluate.sh` |
| `EPOCHS`               | maximum number of epochs for training                        | `start_finetune.sh, evaluate.sh`         |
| `EVAL_BSIZE`           | batch size for evaluation                                    | `evaluate.sh`                            |
| `FAILMAIL`             | send mail to this address if something failed                | `evaluate.sh, start_finetune.sh`         |
| `FOV_DEG`              | field of view in deg. (where FOV relates to the full height) | `start_finetune.sh`                      |
| `INFERENCE_CHKPNT`     | path to checkpoint for inference or training                 | `run.sh`                                 |
| `LR`                   | learning rate                                                | `start_finetune.sh, evaluate.sh`         |
| `MASK_FULL_RES`        | path to full resolution mask                                 | `run.sh, start_finetune.sh, evaluate.sh` |
| `MASK_LUT_DIR`         | path to the directory storing masks and OmniGlasses-LUTs     | `run.sh, start_finetune.sh, evaluate.sh` |
| `MAX_DISP`             | maximum disparity index neglecting residuals                 | `run.sh, start_finetune.sh, evaluate.sh` |
| `NUM_LOADER_WORKER`    | number of dataloader workers (processes)                     | `run.sh, start_finetune.sh, evaluate.sh` |
| `SAVE_PATH`            | directory for output files                                   | `run.sh, start_finetune.sh, evaluate.sh` |
| `SPN_START`            | index of epoch in which SPN module should be started (run.sh: SPN always active) | `run.sh, start_finetune.sh, evaluate.sh` |
| `TEST_BSIZE`           | batch size for testing and validation (training pipeline)    | `start_finetune.sh`                      |
| `TRAIN_BSIZE`          | batch size for training                                      | `start_finetune.sh`                      |

Attention: The baseline is hardcoded in `evaluate.sh`

## References

[1] J. B. Seuffert, A. C. P. Grassi, H. Ahmed, R. Seidel, and G. Hirtz, “OmniGlasses: An Optical Aid for Stereo Vision CNNs to Enable Omnidirectional Image Processing,” In Review, preprint, Apr. 2023. doi: 10.21203/rs.3.rs-2776786/v1.  

[2] Y. Wang et al., “Anytime Stereo Image Depth Estimation on Mobile Devices,” in 2019 International Conference on Robotics and Automation (ICRA), Montreal, QC, Canada: IEEE, 2019, pp. 5893–5900. doi: 10.1109/ICRA.2019.8794003.

[3] J. B. Seuffert, A. C. Perez Grassi, T. Scheck, and G. Hirtz, “A Study on the Influence of Omnidirectional Distortion on CNN-based Stereo Vision,” in Proceedings of the 16th International Joint Conference on Computer Vision, Imaging and Computer Graphics Theory and Applications, VISIGRAPP 2021, Volume 5: VISAPP, Online Conference: SciTePress, Feb. 2021, pp. 809–816. doi: 10.5220/0010324808090816.


## BibTex

```.bibtex
@techreport{seuffert_omniglasses_2023,
    type={preprint},
    title={{OmniGlasses}: {An} {Optical} {Aid} for {Stereo} {Vision} {CNNs} to {Enable} {Omnidirectional} {Image} {Processing}},
    shorttitle={{OmniGlasses}},
    institution={In Review},
    author={Seuffert, Julian Bruno and Perez Grassi, Ana Cecilia and Ahmed, Hamza and Seidel, Roman and Hirtz, Gangolf},
    month=apr,
    year={2023},
    doi={10.21203/rs.3.rs-2776786/v1}
}

@inproceedings{wang_anytime_2019,
    author={Wang, Yan and Lai, Zihang and Huang, Gao and Wang, Brian H. and van der Maaten, Laurens and Campbell, Mark and Weinberger, Kilian Q.},
    booktitle={2019 International Conference on Robotics and Automation (ICRA)}, 
    title={Anytime Stereo Image Depth Estimation on Mobile Devices}, 
    year={2019},
    pages={5893-5900},
    doi={10.1109/ICRA.2019.8794003}
}

@inproceedings{seuffert_study_2021,
    title={A {Study} on the {Influence} of {Omnidirectional} {Distortion} on {CNN}-based {Stereo} {Vision}},
    isbn={978-989-758-488-6},
    doi={10.5220/0010324808090816},
    booktitle={Proceedings of the 16th {International} {Joint} {Conference} on {Computer} {Vision}, {Imaging} and {Computer} {Graphics} {Theory} and {Applications}, {VISIGRAPP} 2021, {Volume} 5: {VISAPP}},
    publisher={SciTePress},
    author={Seuffert, Julian Bruno and Perez Grassi, Ana Cecilia and Scheck, Tobias and Hirtz, Gangolf},
    month=feb,
    year={2021},
    pages={809--816}
}
```
