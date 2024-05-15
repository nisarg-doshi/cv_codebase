import os
import sys

# Set the HOME directory
HOME = os.getcwd()
HOME = os.path.join("/home/jupyter", HOME)
os.chdir(HOME)
print(f"You selected {HOME} directory as your working directory")

# Clone GroundingDINO repository
os.system("git clone https://github.com/IDEA-Research/GroundingDINO.git")
dinno_dir = os.path.join(HOME, "GroundingDINO")
os.chdir(dinno_dir)
print("-----------------------Now you are in the GroundingDINO directory------------------------------")

# Checkout a specific commit
os.system("git checkout -q 57535c5a79791cb76e36fdb64975271354f10251")

# Install GroundingDINO
os.system("pip install -q -e .")

# Move back to HOME directory
os.chdir(HOME)
print("--------------------Now you are in the HOME (current working) directory-----------------------")

# Install segment-anything package
os.system(f"{sys.executable} -m pip install git+https://github.com/facebookresearch/segment-anything.git")

# Create a directory for weights
os.mkdir('weights')
weights_dir = os.path.join(HOME, "weights")
os.chdir(weights_dir)
print("------------------------Now you are in the Weights directory---------------------------------")

# Download weights for GroundingDINO and SAM
os.system("wget -q https://github.com/IDEA-Research/GroundingDINO/releases/download/v0.1.0-alpha/groundingdino_swint_ogc.pth")
os.system("wget -q https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth")
print("-------------------------Downloaded the weights successfully-------------------------------")

# Move back to HOME directory
os.chdir(HOME)

# Verifying the Installation
print("---------------------------------Verifying the Installation-------------------------")

# Check if GroundingDINO weights are downloaded properly
GROUNDING_DINO_CHECKPOINT_PATH = os.path.join(weights_dir, "groundingdino_swint_ogc.pth")
if os.path.isfile(GROUNDING_DINO_CHECKPOINT_PATH):
    print(GROUNDING_DINO_CHECKPOINT_PATH, "; exist:", os.path.isfile(GROUNDING_DINO_CHECKPOINT_PATH))
    print("--------------------Grounding DINO installation is Successful-----------------------------")
else:
    print("Grounding DINO weights are not downloaded properly")

# Check if SAM weights are downloaded properly
SAM_CHECKPOINT_PATH = os.path.join(weights_dir, "sam_vit_h_4b8939.pth")
if os.path.isfile(SAM_CHECKPOINT_PATH):
    print(SAM_CHECKPOINT_PATH, "; exist:", os.path.isfile(SAM_CHECKPOINT_PATH))
    print("--------------------SAM weights downloaded successfully-----------------------------")
else:
    print("SAM weights are not downloaded properly")

# Check if GroundingDINO config file exists
GROUNDING_DINO_CONFIG_PATH = os.path.join(HOME, "GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py")
if os.path.isfile(GROUNDING_DINO_CONFIG_PATH):
    print(GROUNDING_DINO_CONFIG_PATH, "; exist:", os.path.isfile(GROUNDING_DINO_CONFIG_PATH))
    print("--------------------Grounding DINO config file downloaded successfully-----------------------------")
else:
    print("There is something wrong while installing Grounding DINO")
