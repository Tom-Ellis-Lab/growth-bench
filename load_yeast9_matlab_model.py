import cobra
import datetime
from etfl.io import json

from temp_tools import matlab_model_gateway, temp_gateway


if __name__ == "__main__":
    """
    This script loads a yeast9 model from a .mat file and saves it as a JSON file
    .mat file comes from the yeast-GEM repository: https://github.com/SysBioChalmers/yeast-GEM/tree/main/model
    
    The script is used to test the gateway for matlab models
    """
    matlab_model_path = "data/models/yeast9/yeast-GEM.mat"

    # Load the model
    result_model = matlab_model_gateway.load_model_from_mat(file_path=matlab_model_path) # Refactored gateway
    # result_model = temp_gateway.load_matlab_model(infile_path=matlab_model_path) # Original gateway

    if isinstance(result_model, cobra.Model):
        print("Model loaded successfully")

        time = datetime.datetime.now().strftime("%Y%m%d_%H%M")
        if result_model.name is None:
            result_model.name = "yeast9"
        filepath = "data/models/yeast9/{}_{}".format(result_model.name, time)
        # json.save_json_model(result_model, filepath)
        cobra.io.save_json_model(model=result_model, filename=filepath)
