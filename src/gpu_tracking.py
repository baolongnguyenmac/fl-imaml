import nvidia_smi
from time import sleep

def track_gpu(sec:int):
    nvidia_smi.nvmlInit()

    while True:
        handle = nvidia_smi.nvmlDeviceGetHandleByIndex(0)
        info = nvidia_smi.nvmlDeviceGetMemoryInfo(handle)
        name = nvidia_smi.nvmlDeviceGetName(handle)
        percentage_used = info.used/info.total*100
        print(f'{name}: Using {percentage_used:.3f}% ({info.used/(2**30):.3f} GB)')
        # s = f"Device 0: {nvidia_smi.nvmlDeviceGetName(handle)}, Memory : ({100*info.free/info.total:.2f}% free): {info.total}(total), {info.free} (free), {info.used} (used)"
        # print("Device {}: {}, Memory : ({:.2f}% free): {}(total), {} (free), {} (used)".format(0, nvidia_smi.nvmlDeviceGetName(handle), 100*info.free/info.total, info.total, info.free, info.used))
        sleep(sec)

track_gpu(2)
