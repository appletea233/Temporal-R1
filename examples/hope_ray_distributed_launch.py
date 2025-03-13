import os
import json

if "AFO_ENV_CLUSTER_SPEC" in os.environ:
    cluster_spec = json.loads(os.environ["AFO_ENV_CLUSTER_SPEC"])
    afo_spec = json.loads(os.environ["AFO_SPEC"])
    role = cluster_spec["role"]
    assert role == "worker", "{} vs worker".format(role)
    node_rank = cluster_spec["index"]
    # print(cluster_spec, afo_spec)
    
    nnodes = len(cluster_spec[role])
    nproc_per_node = os.popen("nvidia-smi --list-gpus | wc -l").read().strip()
    master = afo_spec['cluster'][role][0]
    master_addr, master_ports = master.split(":")
    master_ports = master_ports.split(",")
    if node_rank == '0':
        ret = f"ray start --head --port={master_ports[0]}"
    else:
        ret = f"ray start --address={master_addr}:{master_ports[0]}"
    print(ret)
else:
    # in case of hope workbench debugging
    nproc_per_node = 2
    nnodes = 1
    node_rank = 0
    master_addr = "localhost"
    master_port = "3333"
    print(
        "torchrun "
        "--nproc_per_node={} "
        "--nnodes={} "
        "--node_rank={} "
        "--master_addr={} "
        "--master_port={}".format(
            nproc_per_node, nnodes, node_rank, master_addr, master_port
        )
    )