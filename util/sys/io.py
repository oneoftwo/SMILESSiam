import os
import shutil


def overwrite_dir(dir_str):
    """ 
    overwrite directory
    """
    if dir_str[-1] != '/':
        dir_str += '/'
    is_dir = os.path.isdir(dir_str)
    if is_dir:
        is_rm = input(f'! overwrite {dir_str}? (y/n): ')
        if is_rm == 'y':
            shutil.rmtree(dir_str)
        else:
            exit()
    os.mkdir(dir_str)
    return dir_str


if __name__ == '__main__':
    pass 

