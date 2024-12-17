source "$(conda info --base)/etc/profile.d/conda.sh"

ENV_NAME=LDAdam

echo ">>>>> Creating environment \"${ENV_NAME}\""
conda create --name $ENV_NAME python=3.12 -y
conda activate $ENV_NAME

echo ">>>>> Installing LDAdam..."
rm -rf build dist *egg* # delete files from previous installations
pip3 install .
echo ">>>>> Installation completed"

rm -rf build
rm -rf *.egg-info
echo ">>>>> Temporary files cleared"