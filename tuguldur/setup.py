import utils

INPUT_DIR = "REM_Phasic-Tonic/data"
POSTTRIAL = "*posttrial*.mat"


if __name__ == '__main__':
    REM_files = utils.find_REM(utils.find_HPC_files(utils.find_matching_files(input_directory=INPUT_DIR, expression=POSTTRIAL)))
    print(REM_files)
    print(len(REM_files))