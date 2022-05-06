def write_header_to_file(title: str, log_filepath: str):
    with open(log_filepath, "a") as f:
        f.write("\n")
        f.write("*"*20 + "\n")
        f.write(title + "\n")
        f.write("*"*20 + "\n")
        f.write("\n")

def log_to_file(txt: str, log_filepath: str):
    with open(log_filepath, "a") as f:
        f.write(txt + "\n")