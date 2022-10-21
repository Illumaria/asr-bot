import os
import uuid
from typing import Any, Tuple

import grpc
import librosa
import numpy as np
import pyogg
import torch
from src.constants import ALPHABET
from src.features import FilterbankFeatures
from tritonclient.grpc import service_pb2
from tritonclient.grpc.service_pb2_grpc import GRPCInferenceServiceStub

featurizer = FilterbankFeatures()


def connect_to_triton_inference_server(
    url: str, model_name: str, model_version: str
) -> Tuple[GRPCInferenceServiceStub, Any, Any]:
    # create gRPC stub for communicating with the server
    channel = grpc.insecure_channel(url)
    grpc_stub = GRPCInferenceServiceStub(channel)

    # get some model properties that we need for data processing
    metadata_request = service_pb2.ModelMetadataRequest(
        name=model_name, version=model_version
    )
    model_metadata = grpc_stub.ModelMetadata(metadata_request)

    config_request = service_pb2.ModelConfigRequest(
        name=model_name, version=model_version
    )
    model_config = grpc_stub.ModelConfig(config_request)

    return grpc_stub, model_metadata, model_config


def ogg_opus_bytes_to_numpy_array(input_bytes: bytes) -> np.ndarray:
    temp_filename = str(uuid.uuid4())
    with open(temp_filename, "wb") as f:
        f.write(input_bytes)
    input_numpy: np.ndarray = pyogg.OpusFile(f.name).as_array().transpose(1, 0)
    os.remove(temp_filename)
    return input_numpy


def preprocess(numpy_data: np.ndarray) -> np.ndarray:
    input_arr = numpy_data.astype(np.float32, order="C") / 32768.0
    resampled_numpy = librosa.resample(input_arr, orig_sr=48000, target_sr=16000)
    resampled_tensor = torch.as_tensor(resampled_numpy).reshape(1, -1)
    seq_len = torch.as_tensor([resampled_tensor.size(-1)])

    processed_input, _ = featurizer(resampled_tensor, seq_len)
    return processed_input.numpy()


def ctc_greedy_decode(predictions: np.ndarray, alphabet: str = ALPHABET) -> str:
    previous_letter_id = blank_id = len(alphabet) - 1
    transcription = list()
    for letter_index in predictions:
        if previous_letter_id != letter_index != blank_id:
            transcription.append(alphabet[letter_index])
        previous_letter_id = letter_index
    return "".join(transcription)


def postprocess(response: Any) -> str:
    """Post-process response to get the text spoken"""
    if len(response.outputs) != 1:
        raise ValueError(f"Expected 1 output, got {len(response.outputs)}")

    if len(response.raw_output_contents) != 1:
        raise ValueError(
            f"Expected 1 output content, got {len(response.raw_output_contents)}"
        )

    output_arr = np.frombuffer(
        response.raw_output_contents[0], dtype=np.float32
    ).reshape(response.outputs[0].shape)
    character_probabilities = output_arr.squeeze().argmax(axis=-1)

    transcription = ctc_greedy_decode(character_probabilities)
    return transcription
