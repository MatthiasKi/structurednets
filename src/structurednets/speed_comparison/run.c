#include <stdio.h>
#include <time.h>
#include <stdlib.h>
#include <string.h>
#include <stdbool.h>
#include <math.h>

#define LAYER_DESCRIPTION_PATH "layer_description.txt"

// Compile with "gcc -o run run.c"
// Run with "./run"

int get_ints(char* name, int* res){
    FILE * fp;
    char * line = NULL;
    size_t len = 0;
    ssize_t read;
    char * ptr;

    fp = fopen(LAYER_DESCRIPTION_PATH, "r");
    if (fp == NULL){
        printf("ERROR could not find the layer description");
        exit(EXIT_FAILURE);
    }

    int found_ints = 0;
    int line_number = 0;
    int name_line_number = 0;
    while ((read = getline(&line, &len, fp)) != -1) {
        if(strstr(line, name)){
            name_line_number = line_number;
            char * cut_line = strtok(line, "\n");
            char * token = strtok(cut_line, " ");
            token = strtok(NULL, " ");
            int idx = 0;
            while( token != NULL ) {
                res[idx] = strtol(token, &ptr, 10);
                token = strtok(NULL, " ");
                idx++;
                found_ints++;
            }
            break;
        }
        line_number++;
    }

    fclose(fp);
    if (line)
        free(line);

    if (found_ints == 0 || name_line_number < 0){
        printf("Could not find the ints for %s", name);
        exit(EXIT_FAILURE);
    }

    return name_line_number;
}

void read_mat_line(char * name, int line_number, float* dest){
    FILE * fp;
    char * line = NULL;
    size_t len = 0;
    ssize_t read;
    char * tod_ptr;
    char * tol_ptr;

    fp = fopen(LAYER_DESCRIPTION_PATH, "r");
    if (fp == NULL){
        printf("ERROR could not find the layer description");
        exit(EXIT_FAILURE);
    }

    bool found_mat = false;
    while ((read = getline(&line, &len, fp)) != -1) {
        if(strstr(line, name)){
            for(int i=0; i < line_number+1; i++){
                getline(&line, &len, fp);
            }

            char * token = strtok(line, ",");
            int idx = 0;
            while(token != NULL){
                dest[idx] = strtod(token, &tod_ptr);
                token = strtok(NULL, ",");
                idx++;
            }
            
            found_mat = true;
            break;
        }
    }

    fclose(fp);
    if (line)
        free(line);

    if(!found_mat){
        printf("Could not find matrix %s\n", name);
        exit(EXIT_FAILURE);
    }
}

void set_name(char * base_name, int state_i, char * dest){
    char buffer[10];
    strcpy(dest, base_name);
    strcat(dest, "_");
    sprintf(buffer, "%d", state_i);
    strcat(dest, buffer);
}

// This function is only used for debugging
void print_vec(float * vec, int len, char * name){
    printf("%s: ", name);
    for(int i = 0; i <len; i++){
        printf("%f", vec[i]);
        if(i < len-1){
            printf(",");
        }
    }
    printf(";\n");
}

int main()
{
    int total_iterations = 1000;

    double time1, time2;
    int input_index, output_index;
    int res[2];

    get_ints("input_size", res);
    int input_size = res[0];

    get_ints("output_size", res);
    int output_size = res[0];

    float u[input_size];
    float y[output_size];

    float checksum_inp[input_size];
    read_mat_line("checksum_inp", 0, checksum_inp);

    // Load the Standard Layer
    float W[output_size][input_size];
    for(int i=0; i<output_size; i++){
        read_mat_line("W", i, W[i]);
    }

    float standard_bias[output_size];
    read_mat_line("standard_bias", 0, standard_bias);

    float standard_checksum_ground_truth[output_size];
    read_mat_line("standard_checksum_out", 0, standard_checksum_ground_truth);
    // ---

    // Standard Layer Time Measurement
    time1 = (double)clock();
    for (int iter = 0; iter < total_iterations; iter++)
    {
        // y = standard_bias + W * u
        for (int i = 0; i < output_size; i++)
        {
            y[i] = standard_bias[i];
            for (int j = 0; j < input_size; j++)
            {
                y[i] += W[i][j] * checksum_inp[j];
            }
        }
    }
    time2 = (double)clock();
    printf("dense_time: %fms\n", (time2 - time1) * 1000 / CLOCKS_PER_SEC / total_iterations);
    // ---

    // Checksum Test for the Standard Layer
    for (int i = 0; i < output_size; i++)
    {
        if(fabs(y[i] - standard_checksum_ground_truth[i]) > 1e-3){
            printf("ERROR: Checksum mismatch for the standard layer in dimension %d\n", i);
            printf("Standard checksum ground truth: %f\n", standard_checksum_ground_truth[i]);
            printf("Computed Y value: %f\n", y[i]);
            printf("Absolute Error: %f\n", fabs(y[i] - standard_checksum_ground_truth[i]));
            exit(EXIT_FAILURE);
        }
    }
    // ---

    // Load the SSS Layer
    get_ints("nb_states", res);
    int nb_states = res[0];

    get_ints("max_state_space_dim", res);
    int max_state_space_dim = res[0];

    get_ints("max_input_dim", res);
    int max_input_dim = res[0];

    get_ints("max_output_dim", res);
    int max_output_dim = res[0];

    int input_dimensions[nb_states];
    int output_dimensions[nb_states];
    int causal_state_dimensions[nb_states];
    int anticausal_state_dimensions[nb_states];

    float A[nb_states][max_state_space_dim][max_state_space_dim];
    float B[nb_states][max_state_space_dim][max_input_dim];
    float C[nb_states][max_output_dim][max_state_space_dim];
    float D[nb_states][max_output_dim][max_input_dim];
    float E[nb_states][max_state_space_dim][max_state_space_dim];
    float F[nb_states][max_state_space_dim][max_input_dim];
    float G[nb_states][max_output_dim][max_state_space_dim];

    float sss_bias[output_size];
    read_mat_line("sss_bias", 0, sss_bias);

    char name[10];
    for(int state_i=0; state_i < nb_states; state_i++){
        set_name("B", state_i, name);
        get_ints(name, res);
        causal_state_dimensions[state_i] = res[0];
        input_dimensions[state_i] = res[1];

        set_name("D", state_i, name);
        get_ints(name, res);
        output_dimensions[state_i] = res[0];

        set_name("E", state_i, name);
        get_ints(name, res);
        anticausal_state_dimensions[state_i] = res[0];

        for(int i=0; i<causal_state_dimensions[state_i]; i++){
            set_name("A", state_i, name);
            read_mat_line(name, i, A[state_i][i]);
            set_name("B", state_i, name);
            read_mat_line(name, i, B[state_i][i]);
        }

        for(int i=0; i<anticausal_state_dimensions[state_i]; i++){
            set_name("E", state_i, name);
            read_mat_line(name, i, E[state_i][i]);
            set_name("F", state_i, name);
            read_mat_line(name, i, F[state_i][i]);
        }

        for(int i=0; i<output_dimensions[state_i]; i++){
            set_name("C", state_i, name);
            read_mat_line(name, i, C[state_i][i]);
            set_name("D", state_i, name);
            read_mat_line(name, i, D[state_i][i]);
            set_name("G", state_i, name);
            read_mat_line(name, i, G[state_i][i]);
        }
    }

    float x1[max_state_space_dim];
    float x1_next[max_state_space_dim];
    float x2[max_state_space_dim];
    float x2_next[max_state_space_dim];

    float sss_checksum_ground_truth[output_size];
    read_mat_line("sss_checksum_out", 0, sss_checksum_ground_truth);
    // ---

    // SSS Layer Time Measurement
    time1 = (double)clock();
    for (int iter = 0; iter < total_iterations; iter++)
    {
        // Reset states
        for(int i=0; i<max_state_space_dim; i++){
            x1[i] = 0;
            x2[i] = 0;
        }

        // Set outputs to the bias of the layer (sparing one addition in the end)
        for(int i=0; i<output_size; i++){
            y[i] = sss_bias[i];
        }

        // Forward loop as in the Python Semiseparable Layer equivalent
        int causal_input_idx = 0;
        int causal_output_idx = 0;
        int anticausal_input_idx = input_size;
        int anticausal_output_idx = output_size;
        for(int causal_state_i=0; causal_state_i < nb_states; causal_state_i++){
            int prev_causal_state_size = 0;
            if(causal_state_i > 0){
                prev_causal_state_size = causal_state_dimensions[causal_state_i-1];
            }
            int curr_causal_state_size = causal_state_dimensions[causal_state_i];

            int anti_causal_state_i = nb_states-1-causal_state_i;
            int prev_anticausal_state_size = 0;
            if(anti_causal_state_i < nb_states - 1){
                prev_anticausal_state_size = anticausal_state_dimensions[anti_causal_state_i+1];
            }
            int curr_anticausal_state_size = anticausal_state_dimensions[anti_causal_state_i];

            for(int i=0; i<prev_causal_state_size; i++){
                x1[i] = x1_next[i];
            }
            for(int i=0; i<prev_anticausal_state_size; i++){
                x2[i] = x2_next[i];
            }

            for(int out_offset=0; out_offset<output_dimensions[causal_state_i]; out_offset++){
                int out_idx = causal_output_idx + out_offset;
                for(int state_offset=0; state_offset<prev_causal_state_size; state_offset++){
                    y[out_idx] += C[causal_state_i][out_offset][state_offset] * x1[state_offset];
                }
                for(int inp_offset=0; inp_offset<input_dimensions[causal_state_i]; inp_offset++){
                    y[out_idx] += D[causal_state_i][out_offset][inp_offset] * checksum_inp[causal_input_idx+inp_offset];
                }
            }

            for(int out_offset=0; out_offset<output_dimensions[anti_causal_state_i]; out_offset++){
                int out_idx = anticausal_output_idx - out_offset - 1;
                for(int state_offset=0; state_offset<prev_anticausal_state_size; state_offset++){
                    y[out_idx] += G[anti_causal_state_i][output_dimensions[anti_causal_state_i] - 1 - out_offset][state_offset] * x2[state_offset];
                }
            }

            for(int state_offset=0; state_offset<curr_causal_state_size; state_offset++){
                x1_next[state_offset] = 0;
                for(int prev_state_offset=0; prev_state_offset<prev_causal_state_size; prev_state_offset++){
                    x1_next[state_offset] += A[causal_state_i][state_offset][prev_state_offset] * x1[prev_state_offset];
                }

                for(int inp_offset=0; inp_offset < input_dimensions[causal_state_i]; inp_offset++){
                    x1_next[state_offset] += B[causal_state_i][state_offset][inp_offset] * checksum_inp[causal_input_idx+inp_offset];
                }
            }

            for(int state_offset=0; state_offset<curr_anticausal_state_size; state_offset++){
                x2_next[state_offset] = 0;
                for(int prev_state_offset=0; prev_state_offset<prev_anticausal_state_size; prev_state_offset++){
                    x2_next[state_offset] += E[anti_causal_state_i][state_offset][prev_state_offset] * x2[prev_state_offset];
                }

                for(int inp_offset=0; inp_offset < input_dimensions[anti_causal_state_i]; inp_offset++){
                    x2_next[state_offset] += F[anti_causal_state_i][state_offset][inp_offset] * checksum_inp[anticausal_input_idx - input_dimensions[anti_causal_state_i] + inp_offset];
                }
            }

            causal_input_idx += input_dimensions[causal_state_i];
            causal_output_idx += output_dimensions[causal_state_i];
            anticausal_input_idx -= input_dimensions[anti_causal_state_i];
            anticausal_output_idx -= output_dimensions[anti_causal_state_i];
        }
    }
    time2 = (double)clock();
    printf("sss_time: %fms \n", (time2 - time1) * 1000 / CLOCKS_PER_SEC / total_iterations);
    // ---

    // Checksum Test for the SSS Layer
    for (int i = 0; i < output_size; i++)
    {
        if(fabs(y[i] - sss_checksum_ground_truth[i]) > 1e-3){
            printf("ERROR: Checksum mismatch for the SSS layer in dimension %d\n", i);
            printf("SSS checksum ground truth: %f\n", sss_checksum_ground_truth[i]);
            printf("Computed Y value: %f\n", y[i]);
            printf("Absolute Error: %f\n", fabs(y[i] - sss_checksum_ground_truth[i]));
            exit(EXIT_FAILURE);
        }
    }
    // ---
    
    return 0;
}

