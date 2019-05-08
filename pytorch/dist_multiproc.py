import sys
import subprocess
import json
import os

def host_to_rank(hostname):
    if hostname == "eco-11":
        return 0
    if hostname == "eco-12":
        return 1
    if hostname == "eco-13":
        return 2
    if hostname == "eco-14":
        return 3

def main():
    argslist = list(sys.argv)[1:]
    configs = {}
    with open("dist_config.txt") as configfile:
        for line in configfile.readlines():
            key = line.split(":")[0]
            value = line.split(":")[1].strip()
            configs[key] = value
    world_size = int(configs["world_size"])
    gpus = int(configs["gpus"])
    master_addr = configs["master_addr"]
    master_port = configs["master_port"]
    nccl_ifname = configs["nccl_ifname"]

    os.environ["MASTER_ADDR"] = master_addr
    os.environ["MASTER_PORT"] = master_port
    os.environ["NCCL_SOCKET_IFNAME"] = nccl_ifname
    os.environ["NCCL_DEBUG"] = "INFO"
    os.environ["NCCL_BUFFSIZE"] = str(16 * 1024 * 1024)

    node_rank = host_to_rank(os.environ["HOSTNAME"])
    workers = []

    for i in range(gpus):
        if '--rank' in argslist:
            argslist[argslist.index('--rank') + 1] = str(node_rank * gpus + i)
        else:
            argslist.append('--rank')
            argslist.append(str(node_rank * gpus + i))
        if '--gpu-rank' in argslist:
            argslist[argslist.index('--gpu-rank') + 1] = str(i)
        else:
            argslist.append('--gpu-rank')
            argslist.append(str(i))
        if '--world-size' in argslist:
            argslist[argslist.index('--world-size') + 1] = str(world_size)
        else:
            argslist.append('--world-size')
            argslist.append(str(world_size))

        stdout = None if i == 0 else open("GPU_" + str(i) + ".log", "w")
        worker = subprocess.Popen([str(sys.executable)] + argslist, stdout=stdout)
        workers.append(worker)

    returncode = 0
    try:
        for worker in workers:
            worker_returncode = worker.wait()
            if worker_returncode != 0:
                returncode = 1
    except KeyboardInterrupt:
        print('Pressed CTRL-C, TERMINATING')
        for worker in workers:
            worker.terminate()
        for worker in workers:
            worker.wait()
        raise

    sys.exit(returncode)


if __name__ == "__main__":
    main()

