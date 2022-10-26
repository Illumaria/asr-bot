# Automatic Speech Bot
Telegram bot for Automatic Speech Recognition

## Prerequisites

* [docker-compose](https://docs.docker.com/compose/install/) >= 2.10.2

## Usage

1. Set the `API_TOKEN` environment variable.

    For Linux:

    ```shell
    export API_TOKEN=<your_api_token>
    ```
    For Windows PowerShell:
    
    ```shell
    $env:API_TOKEN="<your_api_token>"
    ```

2. Run the following command:

    ```shell
    docker-compose up --build -d
    ```

3. Now try the bot by the link http://t.me/another_asr_bot!

### About the Model

The [link](https://ngc.nvidia.com/catalog/models/nvidia:nemospeechmodels) to the up-to-date `Quartznet15x5Base-En.nemo` model was found [here](https://docs.nvidia.com/deeplearning/nemo/user-guide/docs/en/stable/asr/results.html#speech-recognition-languages). The `.nemo` format can then be converted to `.onnx` by using the following commands (assuming `nemo-toolkit[all]` is installed in the current environment):

```commandline
wget https://api.ngc.nvidia.com/v2/models/nvidia/nemospeechmodels/versions/1.0.0a5/files/QuartzNet15x5Base-En.nemo
wget https://raw.githubusercontent.com/NVIDIA/NeMo/main/scripts/export.py
python export.py --autocast --runtime-check --device cpu QuartzNet15x5Base-En.nemo model.onnx
```

The result of such a conversion is already on the `models/quartznet15x5/1/model.onnx` path.
