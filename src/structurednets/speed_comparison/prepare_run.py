import torch 
import numpy as np
import os
import pickle

from structurednets.models.alexnet import AlexNet
from structurednets.asset_helpers import get_animal_classes_filepath
from structurednets.features.extract_features import get_required_indices

def print_mat_list_to_file(mats: list, name: str, outfile_path: str):
    for k, mat in enumerate(mats):
        print_mat_to_file(mat=mat, name=f"{name}_{k}", outfile_path=outfile_path)

def print_mat_to_file(mat: torch.tensor, name: str, outfile_path: str):
    with open(outfile_path, "a") as f:
        f.write(f"{name} {mat.shape[0]} {mat.shape[1]}\n")
        for i in range(mat.shape[0]):
            for j in range(mat.shape[1]):
                f.write(f"{mat[i][j]}")
                if j < mat.shape[1] - 1:
                    f.write(",")
            f.write("\n")

def print_vec_to_file(vec: torch.tensor, name: str, outfile_path: str):
    with open(outfile_path, "a") as f:
        f.write(f"{name} {len(vec)}\n")
        for i in range(len(vec)):
            f.write(f"{vec[i]}")
            if i < len(vec) - 1:
                f.write(",")
        f.write("\n")

def prepare_run(model_path: str, outfile_path="layer_description.txt"):
    label_filepath = get_animal_classes_filepath()
    required_indices = get_required_indices(path_to_label_file=label_filepath)
    model = AlexNet(output_indices=required_indices, use_gpu=False)

    optim_mat_t = model.get_optimization_matrix().T
    optim_mat = optim_mat_t.cpu().detach().numpy()
    bias_t = model.get_bias()
    bias = bias_t.cpu().detach().numpy()

    input_size = optim_mat.shape[1]
    output_size = optim_mat.shape[0]

    checksum_inp = np.random.uniform(-1,1,size=(1,input_size))
    checksum_inp_t = torch.tensor(checksum_inp).float()

    standard_checksum_output = torch.squeeze(torch.matmul(optim_mat_t, checksum_inp_t.T)) + bias_t

    res_dict = pickle.load(open(model_path, "rb"))
    nb_states = res_dict["nb_states"]
    sss_layer = res_dict["model"][0]
    sss_checksum_output = sss_layer.forward(checksum_inp_t)

    with open(outfile_path, "w") as f:
        f.write(f"input_size {input_size}\n")
        f.write(f"output_size {output_size}\n")

    # Checksum used by both layers
    print_mat_to_file(mat=checksum_inp_t, name="checksum_inp", outfile_path=outfile_path)

    # Original layer
    print_mat_to_file(mat=optim_mat, name="W", outfile_path=outfile_path)
    print_vec_to_file(vec=bias, name="standard_bias", outfile_path=outfile_path)
    print_vec_to_file(vec=standard_checksum_output, name="standard_checksum_out", outfile_path=outfile_path)

    # SSS layer
    with open(outfile_path, "a") as f:
        f.write(f"nb_states {nb_states}\n")
        f.write(f"max_state_space_dim {max([max(A_k.shape) for A_k in sss_layer.A])}\n")
        f.write(f"max_input_dim {max([B_k.shape[1] for B_k in sss_layer.B])}\n")
        f.write(f"max_output_dim {max([D_k.shape[0] for D_k in sss_layer.D])}\n")
    print_mat_list_to_file(mats=sss_layer.A, name="A", outfile_path=outfile_path)
    print_mat_list_to_file(mats=sss_layer.B, name="B", outfile_path=outfile_path)
    print_mat_list_to_file(mats=sss_layer.C, name="C", outfile_path=outfile_path)
    print_mat_list_to_file(mats=sss_layer.D, name="D", outfile_path=outfile_path)
    print_mat_list_to_file(mats=sss_layer.E, name="E", outfile_path=outfile_path)
    print_mat_list_to_file(mats=sss_layer.F, name="F", outfile_path=outfile_path)
    print_mat_list_to_file(mats=sss_layer.G, name="G", outfile_path=outfile_path)
    print_vec_to_file(vec=sss_layer.bias, name="sss_bias", outfile_path=outfile_path)
    print_mat_to_file(mat=sss_checksum_output, name="sss_checksum_out", outfile_path=outfile_path)

def execute_run():
    folder_path = os.path.dirname(os.path.realpath(__file__))
    runscript_path = os.path.join(folder_path, "run")
    stream = os.popen(runscript_path)
    output = stream.read()

    dense_time_string = output.split("\n")[0]
    sss_time_string = output.split("\n")[1]

    dense_time = float(dense_time_string.split(" ")[-1].split("ms")[0])
    sss_time = float(sss_time_string.split(" ")[1].split("ms")[0])

    return dense_time, sss_time

if __name__ == "__main__":
    param_shares = [0.01, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3] #[0.01, 0.1, 0.2, 0.3, 0.4, 0.5]
    nb_iterations = 5
    base_model_path = "/path/to/speed_experiments_2/"
    output_filepath = "/path/to/speed_experiments_2/speed_results.csv"

    with open(output_filepath, "w") as output_file:
        output_file.write("ParamShare,Iteration,DenseTime,SSSTime\n")

        for param_share in param_shares:
            for iteration in range(nb_iterations):
                model_path = os.path.join(base_model_path, f"InceptionV3_animal_sss_finetune_res_lr_0.001_batch_size_64_paramshare_{param_share}_states_50_iter_{iteration}.p")

                prepare_run(model_path=model_path)
                print("Executing Speed Comparison for " + model_path)
                dense_time, sss_time = execute_run()

                output_file.write(f"{param_share},{iteration},{dense_time},{sss_time}\n")