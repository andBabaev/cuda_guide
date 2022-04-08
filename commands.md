virtualenv ~/python_env/cuda_guide
source ~/python_env/cuda_guide/bin/activate

pip install pip -U


docker build -t cuda_guide -f Dockerfile.app.torch .
docker rm --force cuda_guide
docker run -p 8501:8501 --gpus 0 --name cuda_guide -it cuda_guide:latest
docker exec -it cuda_guide bash
jupyter notebook --allow-root --port 9999

docker build -t cuda_guide_nvidia -f Dockerfile.app.
docker rm --force cuda_guide_nvidia
docker run -p 8501:8501 --gpus 0 --name cuda_guide_nvidia -it cuda_guide_nvidia:latest
docker exec -it cuda_guide_nvidia bash


ssh -i cuda-sweft.pem ubuntu@109.248.175.39


sudo apt update && sudo apt upgrade -y
sudo add-apt-repository ppa:graphics-drivers/ppa -y
sudo apt update
sudo apt install nvidia-driver-510 -y
sudo reboot
nvidia-smi


sudo apt-get update
sudo apt-get install ca-certificates curl gnupg lsb-release
echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/docker-archive-keyring.gpg] https://download.docker.com/linux/ubuntu $  (lsb_release -cs) stable" | sudo tee /etc/apt/sources.list.d/docker.list > /dev/null
sudo apt-get update
sudo apt-get install docker-ce docker-ce-cli containerd.io
sudo usermod -aG docker ${USER}
su - ${USER}

docker build -t cuda_guide -f Dockerfile.app.torch .
docker run -p 8501:8501 --gpus 0 --name cuda_guide -it cuda_guide:latest