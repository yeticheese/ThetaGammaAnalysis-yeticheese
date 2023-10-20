import utils

INPUT_DIR = "REM_Phasic-Tonic/data"
POSTTRIAL = "*posttrial*.mat"

if __name__ == '__main__':
    HPC_files = utils.find_HPC_files(utils.find_matching_files(input_directory=INPUT_DIR, expression=POSTTRIAL))
    utils.extract_REM(HPC_files)