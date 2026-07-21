import os
import subprocess
import time

def D_LSNARS():

    V3DPath = "Your vaa3d path /v3d_external/bin/vaa3d_msvc.exe"
    pluginName = "Your vaa3d path  /vaa3d_tools/bin/plugins/D_LSNARS/single_neuron_reconstruction\single_neuron_reconstruction.dll"
    funcName = "reconstruction"

    pytorch_path = "Your envs path"
    python_configuration_path = 'Your code download path/setup.yaml'

    image_path = ""
    marker_files = " "
    output_path =  " "

    marker_list = os.listdir(marker_files)

    time_log_path = os.path.join(marker_files, "time_log.txt")
    if (not os.path.exists(time_log_path)) or os.path.getsize(time_log_path) == 0:
        with open(time_log_path, "w", encoding="utf-8") as f:
            f.write("marker_name \t time_hours\n")

    for marker in marker_list:
        if marker.endswith('txt'):
            continue

        marker_path = os.path.join(marker_files, marker)
        cmd = [V3DPath,
               '/x', pluginName,
               '/f', funcName,
               '/i', image_path, marker_path, pytorch_path,
               '/o', output_path,
               '/p', python_configuration_path,

               ]

        start_time = time.time()

        try:
            result = subprocess.run(
                cmd,
                stderr=subprocess.PIPE,
                check=False
            )
            # break
            return_code = result.returncode

            if result.returncode != 0:
                print("子程序运行失败，但继续执行后续代码，返回码:", result.returncode)

        except Exception as e:
            print("子程序执行异常，已跳过:", repr(e))
            return_code = -1

        end_time = time.time()
        cost_time = (end_time - start_time) / 3600.0

        print(f"{marker} finished, time: {cost_time:.4f} h, return_code: {return_code}")

        with open(time_log_path, "a", encoding="utf-8") as f:
            f.write(f"{marker} \t {cost_time:.4f} \n")

if __name__ == '__main__':
    D_LSNARS()
