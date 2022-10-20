# Automatic Speech Bot
Telegram bot for Automatic Speech Recognition

## Usage

### Get the Model

The [link](https://ngc.nvidia.com/catalog/models/nvidia:nemospeechmodels) to the up-to-date `Quartznet15x5Base-En.nemo` model was found [here](https://docs.nvidia.com/deeplearning/nemo/user-guide/docs/en/stable/asr/results.html#speech-recognition-languages). The `.nemo` format can then converted to `.onnx` by using the following commands (assuming `nemo-toolkit[all]` is installed in the current environment):

```commandline
wget https://api.ngc.nvidia.com/v2/models/nvidia/nemospeechmodels/versions/1.0.0a5/files/QuartzNet15x5Base-En.nemo
wget https://raw.githubusercontent.com/NVIDIA/NeMo/main/scripts/export.py
python export.py --autocast --runtime-check --device cpu QuartzNet15x5Base-En.nemo model.onnx
```

The result of such a conversion is already on the `models/quartznet15x5/1/model.onnx` path.

### Run Triton Inference Server

```commandline
docker pull nvcr.io/nvidia/tritonserver:21.09-py3
docker run --rm --name triton --net bridge --add-host=host.docker.internal:host-gateway -p 8000:8000 -p 8001:8001 -p 8002:8002 -v $(pwd)/models:/models nvcr.io/nvidia/tritonserver:21.09-py3 tritonserver --model-repository=/models --strict-model-config=false 
```

### Run Backend Server

```commandline
docker build --rm -t asr-backend -f backend/Dockerfile backend
docker run --rm --name asr-backend --net bridge --add-host=host.docker.internal:host-gateway -e URL=host.docker.internal:8001 -p 5000:5000 asr-backend
```

### Run Telegram Bot

Set `API_TOKEN` environment variable.

For Linux:

```commandline
export API_TOKEN=<your_api_token>
```
For Windows PowerShell:

```commandline
$env:API_TOKEN="<your_api_token>"
```

Then run the following commands:

```commandline
pip install -r requirements.txt
python bot.py
```

Now try the bot by the link http://t.me/another_asr_bot!
