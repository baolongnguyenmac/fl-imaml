import nvidia_smi
from time import sleep

def track_gpu(sec:int):
    nvidia_smi.nvmlInit()
    deviceCount = nvidia_smi.nvmlDeviceGetCount()

    while True:

        used = 0
        total = 0

        for i in range(deviceCount):
            handle = nvidia_smi.nvmlDeviceGetHandleByIndex(i)
            info = nvidia_smi.nvmlDeviceGetMemoryInfo(handle)
            total += info.total
            used += info.used
            # name = nvidia_smi.nvmlDeviceGetName(handle)

        percentage_used = used/total*100
        print(f'{deviceCount} GPU(s): Using {percentage_used:.3f}% ({used/(2**30):.3f} GB)')

        sleep(sec)

track_gpu(1.5)
