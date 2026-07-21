# D-LSNAR

    single_neuron_reconstruction/
    
    ├── CMakeLists.txt
    ├── src
    ├── Multi_neuron_reconstruction.py
    
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
    └── D_LSNARS
            └── Single Neuron Reconstruction
  Then choose the input image and soma marker to perform automatic reconstruction.
  <img width="742" height="551" alt="image" src="https://github.com/user-attachments/assets/f8938a10-b343-4174-a781-96bc55cb5590" />

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


  
