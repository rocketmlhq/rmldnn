import os
import sys
import argparse
import numpy as np
from tqdm import tqdm
from multiprocessing import Pool


def calculate_omega_based_on_eta(eta):
	tol = 1e-6
	if np.abs(eta - 0.1) < tol:
		omega = np.array([
                     2.627675432985797,
                     5.307324799118128,
                     8.067135580679963,
                    10.908707509765620,
                    13.819191590843053,
                    16.782691099052428,
                    19.785505130248573,
                    22.817253043828419,
                    25.870442560222948,
                    28.939736049582585
            ])
	elif np.abs(eta - 0.2) < tol:
		omega = np.array([
                     2.284453709564703,
                     4.761288969346805,
                     7.463676172029721,
                    10.326611007844360,
                    13.286241503970587,
                    16.303128640923813,
                    19.355160454004977,
                    22.429811599309446,
                    25.519693779498752,
                    28.620245932841211
            ])
	elif np.abs(eta - 0.5) < tol:
		omega = np.array([
                     1.720667178038759,
                     4.057515676220868,
                     6.851236918963457,
                     9.826360878869767,
                    12.874596358343892,
                    15.957331424826481,
                    19.058668810723926,
                    22.171076812994045,
                    25.290574447713286,
                    28.414873450382377
            ])
	elif np.abs(eta - 0.7) < tol:
		omega = np.array([
                     1.513246031735345,
                     3.851891808005561,
                     6.703141757332143,
                     9.716730053822916,
                    12.788857060379099,
                    15.887318867290485,
                    18.999652186088099,
                    22.120134252280451,
                    25.245793691314280,
                    28.374941402170549
            ])
	elif np.abs(eta - 1.0) < tol:
		omega = np.array([
                     1.306542374188806,
                     3.673194406304252,
                     6.584620042564173,
                     9.631684635691871,
                    12.723240784131329,
                    15.834105369332415,
                    18.954971410841591,
                    22.081659635942589,
                    25.212026888550827,
                    28.344864149599882
            ])
	return omega


def construct_KL_sum_2D(args, x, y, rand_tensor_list):
    sin = np.sin
    cos = np.cos
    sigma_x = 1
    sigma_y = 1
    eta_x = args.eta_x
    eta_y = args.eta_y

    omega_x = calculate_omega_based_on_eta(args.eta_x)
    omega_y = calculate_omega_based_on_eta(args.eta_y)

    # omegaX omegaY are vectors
    lambda_x = 2.0 * eta_x * sigma_x / (1.0 + (eta_x * omega_x) ** 2);
    lambda_y = 2.0 * eta_y * sigma_y / (1.0 + (eta_y * omega_y) ** 2);
    
    kl_sum = 0*x # 3d

    for i in range(6):
        kl_sum += rand_tensor_list[i] * np.sqrt(lambda_x[i]) * np.sqrt(lambda_y[i]) * (eta_x * omega_x[i] * cos(omega_x[i] * x) + sin(omega_x[i] * x)) * (eta_y * omega_y[i] * cos(omega_y[i] * y) + sin(omega_y[i] * y))   

    return kl_sum

def construct_KL_sum_3D(args, x, y, z, rand_tensor_list):
    sin = np.sin
    cos = np.cos
    sigma_x = 1
    sigma_y = 1
    sigma_z = 1
    eta_x = args.eta_x
    eta_y = args.eta_y
    eta_z = args.eta_z

    omega_x = calculate_omega_based_on_eta(args.eta_x)
    omega_y = calculate_omega_based_on_eta(args.eta_y)
    omega_z = calculate_omega_based_on_eta(args.eta_z)

    # omegas are vectors
    lambda_x = 2.0 * eta_x * sigma_x / (1.0 + (eta_x * omega_x) ** 2);
    lambda_y = 2.0 * eta_y * sigma_y / (1.0 + (eta_y * omega_y) ** 2);
    lambda_z = 2.0 * eta_z * sigma_z / (1.0 + (eta_z * omega_z) ** 2);

    kl_sum = 0*x # 4d

    for i in range(6):
        kl_sum += rand_tensor_list[i] * np.sqrt(lambda_x[i]) * np.sqrt(lambda_y[i]) * np.sqrt(lambda_z[i]) * (eta_x * omega_x[i] * cos(omega_x[i] * x) + sin(omega_x[i] * x)) * (eta_y * omega_y[i] * cos(omega_y[i] * y) + sin(omega_y[i] * y)) * (eta_z * omega_z[i] * cos(omega_z[i] * z) + sin(omega_z[i] * z))

    return kl_sum

def calc_coefficients_nu_logK(args):
    a_min = -3
    a_max = 3
    list_of_coeff_tensors = []

    if args.dims == 2:
        shape = (1, args.output_size, args.output_size)
    else:
        shape = (1, args.output_size, args.output_size, args.output_size)

    for i in range(args.n_sum_nu):        
        coeff_rand = a_min + (a_max - a_min) * np.random.rand()
 
        coeff_repeated = np.repeat(coeff_rand, args.output_size ** args.dims)
        coeff_np = coeff_repeated.reshape(shape)
        list_of_coeff_tensors.append(coeff_np)

    return list_of_coeff_tensors

def grid2D(nx, ny):
    x = np.linspace(0, 1, nx)
    y = np.linspace(0, 1, ny)
    xv, yv = np.meshgrid(x, y)

    return xv, yv

def grid3D(nx, ny, nz):
    x = np.linspace(0, 1, nx)
    y = np.linspace(0, 1, ny)
    z = np.linspace(0, 1, nz)
    xv, yv, zv = np.meshgrid(x, y, z)

    return xv, yv, zv

def generate_diffusivity_tensor(args):
    pi = np.pi
    sin = np.sin
    cos = np.cos
    exp = np.exp

    nx = args.output_size
    ny = args.output_size
    nz = args.output_size

    if args.dims == 2:
        xarr, yarr = grid2D(nx, ny)
        x = np.expand_dims(xarr, 0)
        y = np.expand_dims(yarr, 0)
        shape = (1, nx, ny)
    else:
        xarr, yarr, zarr = grid3D(nx, ny, nz)
        x = np.expand_dims(xarr, 0)
        y = np.expand_dims(yarr, 0)
        z = np.expand_dims(zarr, 0)
        shape = (1, nx, ny, nz)

    # Construct nu
    a1 = np.zeros(shape, dtype=np.float32)
    a2 = np.zeros(shape, dtype=np.float32)
    a3 = np.zeros(shape, dtype=np.float32)
    a4 = np.zeros(shape, dtype=np.float32)
    a5 = np.zeros(shape, dtype=np.float32)
    a6 = np.zeros(shape, dtype=np.float32)

    list_of_coeff_tensors_nu = calc_coefficients_nu_logK(args)

    a1 = list_of_coeff_tensors_nu[0]
    if (args.n_sum_nu > 1):
        a2 = list_of_coeff_tensors_nu[1]
    if (args.n_sum_nu > 2):
        a3 = list_of_coeff_tensors_nu[2]
    if (args.n_sum_nu > 3):
        a4 = list_of_coeff_tensors_nu[3]
    if (args.n_sum_nu > 4):
        a5 = list_of_coeff_tensors_nu[4]
    if (args.n_sum_nu > 5):
        a6 = list_of_coeff_tensors_nu[5]    

    if args.dims == 2:
        nu_tensor = exp(construct_KL_sum_2D(args, x, y, [a1, a2, a3, a4, a5, a6]))
    else:
        nu_tensor = exp(construct_KL_sum_3D(args, x, y, z, [a1, a2, a3, a4, a5, a6]))

    return nu_tensor

def build_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_samples', '-n',  help='Number of files (diffusivity maps) to generate',          type=int,   default=1)
    parser.add_argument('--dir',         '-d',  help='Output (generate mode) or input (convert mode) dir',      default='./diffusivity_maps')
    parser.add_argument('--output_size', '-s',  help='Output size of the generated image (single dimension)',   type=int,   default=64)
    parser.add_argument('--dims',        '-D',  help='Data dimenionality',                                      type=int,   default=2)
    parser.add_argument('--n_sum_nu',           help='Number of terms in nu expression', choices=[1,2,3,4,5,6], type=int,   default=4)
    parser.add_argument('--eta_x',              help='Correlation length in x direction',                       type=float, default=0.5)
    parser.add_argument('--eta_y',              help='Correlation length in y direction',                       type=float, default=0.5)
    parser.add_argument('--eta_z',              help='Correlation length in z direction',                       type=float, default=0.5)
    parser.add_argument('--cores',              help='Number of cores for parallel processing',                 type=int,   default=1)
    return parser

def create_directory(name):
    if os.path.exists(name) and any(os.scandir(name)):
        raise ValueError('Directory not empty: ' + name)
    elif not os.path.exists(name):
        os.makedirs(name)

def generate_nu_tensor(task):
    id   = task[0]
    args = task[1]

    filename = args.dir + "/diffusivity_" + str(id) + ".npy"
    nu_tensor = generate_diffusivity_tensor(args)
    arr = nu_tensor.astype(np.float32)
    np.save(filename, arr)

def main():
    parser = build_parser()
    args = parser.parse_args()

    create_directory(args.dir)
    np.random.seed(42)

    if args.dims < 2 or args.dims > 3:
        raise ValueError('Can only generate 2D or 3D data')

    cores = args.cores
    tasks = [(id, arg) for id, arg in enumerate([args]*args.num_samples)]

    with Pool(cores) as p:
        list(tqdm(p.imap(generate_nu_tensor, tasks), total=args.num_samples))

if __name__ == '__main__':
    print("Command line inputs = ", sys.argv)
    main()
