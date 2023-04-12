import argparse
import os
from mpi4py import MPI
import numpy as np


parser = argparse.ArgumentParser()
parser.add_argument('--blacklight_exe', type=str, help='absolute path to blacklight executable')
parser.add_argument('--input_template', type=str, help='absolute path to the input template')
parser.add_argument('--output_dir', type=str, help='absolute path to the output directory')
parser.add_argument('--frame_iter', type=str, help='iterable specifying the frame indices')
parser.add_argument('--index_format', type=str, default='{0:05d}',
                    help='formatting the indices under the output directory')
parser.add_argument('--num_threads', type=int, default=1, help='number of threads per worker')
parser.add_argument('--config_dir', type=str, default='', help='absolute path to the config files')
parser.add_argument('--num_configs', type=int, default=1, help='number of configs')
parser.add_argument('--no_raytrace', action='store_true')
args = parser.parse_args()

comm = MPI.COMM_WORLD
size = comm.Get_size()
rank = comm.Get_rank()
exec('indices = ' + args.frame_iter)
indices = np.asarray(indices)
num_indices = len(indices)
num_indices_per_rank = num_indices // size
num_indices_extra = num_indices % size
indices_local = list(
    indices[np.arange(rank * num_indices_per_rank, (rank + 1) * num_indices_per_rank)])
if rank < num_indices_extra:
    indices_local = indices_local + [indices[-(rank + 1)]]

assert args.output_dir
try:
    input_dir = os.path.join(args.output_dir, 'input')
    npz_dir =  os.path.join(args.output_dir, 'npz')
    if not os.path.exists(input_dir):
        os.makedirs(input_dir)
    if not os.path.exists(npz_dir):
        os.makedirs(npz_dir)
except FileExistsError:
    pass

for index in indices_local:
    index_format = args.index_format.format(index)
    if args.num_configs > 1:
        for i in range(args.num_configs):
            with open(args.input_template, 'r') as f_in:
                with open(os.path.join(input_dir, index_format + f'.input.{i}'), 'w') as f_out:
                    f_out.write(f_in.read().format(
                        index,
                        os.path.join(npz_dir, index_format + f'.npz.{i}'),
                        args.num_threads,
                        args.config_dir + f'.{i}'
                    ))
            if not args.no_raytrace:
                os.system(args.blacklight_exe + ' ' +
                          os.path.join(input_dir, index_format + f'.input.{i}'))
    elif args.num_configs == 1:
        with open(args.input_template, 'r') as f_in:
            with open(os.path.join(input_dir, index_format + '.input'), 'w') as f_out:
                f_out.write(f_in.read().format(index, os.path.join(npz_dir, index_format + '.npz'),
                                               args.num_threads))
        if not args.no_raytrace:
            os.system(args.blacklight_exe + ' ' + os.path.join(input_dir, index_format + '.input'))
    else:
        raise ValueError('invalid value for args.num_configs.')
