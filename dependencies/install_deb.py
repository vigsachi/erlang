import os
import sys

sys.path.insert(0, os.getcwd())

if __name__ == "__main__":

    # Install debian dependencies.
    os.system('chmod +x dependencies/os/deb/install_req.sh')
    os.system('./dependencies/os/deb/install_req.sh')

    # Install python requirements.
    os.system('pip3 install -y -r requirements.txt')
