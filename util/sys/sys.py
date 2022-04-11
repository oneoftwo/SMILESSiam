import os


def set_cuda_visible_devices(ngpus, is_print=True):
    """ 
    set cuda visible devices environment
    """
    import subprocess
    import os
    import numpy as np
    empty = []
    if ngpus>0:
        fn = f'/tmp/empty_gpu_check_{np.random.randint(0,10000000,1)[0]}'
        for i in range(4):
            os.system(f'nvidia-smi -i {i} | grep "No running" | wc -l > {fn}')
            with open(fn) as f:
                out = int(f.read())
            if int(out)==1:
                empty.append(i)
            if len(empty)==ngpus: break
        if len(empty)<ngpus:
            print ('avaliable gpus are less than required', len(empty), ngpus)
            exit(-1)
        os.system(f'rm -f {fn}')
    cmd = ''
    for i in range(ngpus):        
        cmd+=str(empty[i])+','
    os.environ['CUDA_VISIBLE_DEVICES'] = cmd
    if is_print:
        print(f'cuda device set as {cmd[:-1]}')
    return None


if __name__ == '__main__':
    pass

