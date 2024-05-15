import os
import sys
import argparse


class GroundingDINOInstaller:
    def __init__(self, home_dir):
        self.home_dir = home_dir

# Set the HOME directory
    def set_home_directory(self):
            """
            Sets the HOME directory as the current working directory.
            """
            os.chdir(self.home_dir)
            print(f"You selected {self.home_dir} directory as your working directory")




    # Clone GroundingDINO repository
    def clone_grounding_dino_repository(self):
        """
        Clones the GroundingDINO repository from GitHub.
        """
        os.system("git clone https://github.com/IDEA-Research/GroundingDINO.git")
        dinno_dir = os.path.join(self.home_dir, "GroundingDINO")
        os.chdir(dinno_dir)
        print("-----------------------Now you are in the GroundingDINO directory------------------------------")

    # Checkout a specific commit
    def checkout_specific_commit(self, commit_hash):
        """
        Checks out a specific commit in the GroundingDINO repository.

        Args:
            commit_hash (str): The hash of the commit to checkout.
        """
        os.system(f"git checkout -q {commit_hash}")
        #os.system(f"git checkout -q 57535c5a79791cb76e36fdb64975271354f10251")

    # Install GroundingDINO
    def install_grounding_dino(self):
        """
        Installs the GroundingDINO package.
        """
        os.system("pip install -q -e .")

# Move back to HOME directory


# Move back to HOME directory
    def install_segment_anything_package(self):
        """
        Installs the segment-anything package from the Facebook Research GitHub repository.
        """
        os.chdir(self.home_dir)
        # Install segment-anything package
        os.system(f"{sys.executable} -m pip install git+https://github.com/facebookresearch/segment-anything.git")


    def create_directory_for_weights(self):
        """
        Creates a directory named 'weights' for storing model weights.
        """
        # os.chdir(self.home_dir)
        print("--------------------Now you are in the HOME (current working) directory-----------------------")

                
        # Create a directory for weights
        os.mkdir('weights')
        weights_dir = os.path.join(self.home_dir, "weights")
        os.chdir(weights_dir)
        print("------------------------Now you are in the Weights directory---------------------------------")

        # Download weights for GroundingDINO and SAM
    def download_weights(self):
        """
        Downloads weights for GroundingDINO and SAM models.
        """
        os.system("wget -q https://github.com/IDEA-Research/GroundingDINO/releases/download/v0.1.0-alpha/groundingdino_swint_ogc.pth")
        os.system("wget -q https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth")
        print("-------------------------Downloaded the weights successfully-------------------------------")




# Verifying the Installation
    def verify_installation(self):
        """
        Verifies the installation by checking if downloaded weights and config files exist.
        """
        print("---------------------------------Verifying the Installation-------------------------")
        os.chdir(self.home_dir)

        # Check if GroundingDINO weights are downloaded properly
        GROUNDING_DINO_CHECKPOINT_PATH = os.path.join("weights", "groundingdino_swint_ogc.pth")
        if os.path.isfile(GROUNDING_DINO_CHECKPOINT_PATH):
            print(GROUNDING_DINO_CHECKPOINT_PATH, "; exist:", os.path.isfile(GROUNDING_DINO_CHECKPOINT_PATH))
            print("--------------------Grounding DINO installation is Successful-----------------------------")
        else:
            print("Grounding DINO weights are not downloaded properly")

        # Check if SAM weights are downloaded properly
        SAM_CHECKPOINT_PATH = os.path.join("weights", "sam_vit_h_4b8939.pth")
        if os.path.isfile(SAM_CHECKPOINT_PATH):
            print(SAM_CHECKPOINT_PATH, "; exist:", os.path.isfile(SAM_CHECKPOINT_PATH))
            print("--------------------SAM weights downloaded successfully-----------------------------")
        else:
            print("SAM weights are not downloaded properly")

        # Check if GroundingDINO config file exists
        GROUNDING_DINO_CONFIG_PATH = os.path.join(self.home_dir, "GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py")
        if os.path.isfile(GROUNDING_DINO_CONFIG_PATH):
            print(GROUNDING_DINO_CONFIG_PATH, "; exist:", os.path.isfile(GROUNDING_DINO_CONFIG_PATH))
            print("--------------------Grounding DINO config file downloaded successfully-----------------------------")
        else:
            print("There is something wrong while installing Grounding DINO")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Install GroundingDINO and required dependencies")
    parser.add_argument("--home", type=str, default=os.getcwd(), help="Home directory (default: current working directory)")
    parser.add_argument("--commit", type=str, default="57535c5a79791cb76e36fdb64975271354f10251", help="Commit hash to checkout (default: latest)")
    parser.parse_args()
    args = parser.parse_args()
    installer = GroundingDINOInstaller(args.home)
    installer.set_home_directory()
    installer.clone_grounding_dino_repository()
    installer.checkout_specific_commit(args.commit)
    installer.install_grounding_dino()
    installer.create_directory_for_weights()
    installer.download_weights()
    installer.verify_installation()


