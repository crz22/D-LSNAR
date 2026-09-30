# D-LSNAR

    single_neuron_reconstruction/
    
    ├── CMakeLists.txt
    ├── Multi_neuron_reconstruction.py
    ├── src
      ├── cpp
      ├── python
        ├── setup.yaml
        |── requirements.txt

This example demonstrates how to compile and run D-LSNAR for automated single-neuron reconstruction.

1. Install Dependencies
  Install the following software:
  1)Vaa3D
  2)Visual Studio Code (with CMake/C++ toolchain)
  3)Anaconda/Miniconda
   
2. Compile the Vaa3D Plugin (C++)
Open the C++ project and modify the Vaa3D installation path in CMakeLists.txt:
set(VAA3DPATH "Your/Vaa3D/Path")

Then compile the project using Visual Studio Code (or CMake).
After successful compilation, the generated plugin (.dll) can be placed in the Vaa3D plugin directory.

3. Configure the Python Environment
   1) Create the Python environment:
   conda env create -f requirements.txt

   2) Modify the following parameters in the configuration (.yaml) file:
    Python_code_path: /Your/Download/Path/src/python

    Vaa3d_path: /Your/Vaa3D/v3d_external/bin/vaa3d_msvc.exe

    Vaa3d_resample_plugin_path: /Your/Vaa3D/v3d_external/bin/plugins/neuron_utilities/resample_swc/resample_swc.dll

4. Run D-LSNAR
  Two execution modes are provided.

  Option 1. Run from the Vaa3D GUI (Recommended)
  Open Vaa3D and select
  Plug-ins
    └── D_LSNAR
            └── Single Neuron Reconstruction
  Then choose the input image and soma marker to perform automatic reconstruction.
 <img width="937" height="720" alt="image" src="https://github.com/user-attachments/assets/de9d1b20-356b-4028-ab11-864e79e40ff9" />


  Option 2. Run from Python
  Execute
  python Multi_neuron_reconstruction.py

  Before running, specify
  image path
  soma marker path
  Vaa3D plugin path
  output directory

Output
  The reconstruction pipeline automatically generates reconstructed neuron morphology (.swc)
  intermediate segmentation results (optional)

Test Sample for Quick Validation

  To facilitate quick validation and help users get started, we have provided a test_sample directory in the repository. This sample allows you to reproduce both the local reconstruction and cross-block stitching procedures without requiring a complete whole-brain dataset.
  
  The test_sample includes the following resources:
  
  17302_cut/: Contains 8 adjacent subvolumes (256×256×256 voxels each), saved in the TeraFly format.
  marker/: Contains the soma coordinates of the test neuron in .marker format.
  test_results/: Contains the expected outputs for verification, including:
  17302_cut_tmp/: The intermediate results for each block (e.g., soma mask _somamask.tif, neurite mask _seg.tif, preliminary reconstruction _spe_dnr.swc, and optimized result _resample.swc).
  17302_cut_SPE_DNR.swc: The final globally stitched reconstruction result.
  configuration.yaml: The key parameter settings used to generate these results.
  x15211_y21847_z3895.tif: The stitched whole-volume image of the 8 adjacent subvolumes, provided for visual inspection and validation of the final reconstruction against the raw image data.

You can use the configuration.yaml file provided in the test_results folder to configure your environment and run the test directly.


  
