import struct
from typing import Any, Tuple

import grpc
import numpy as np
from tritonclient.grpc import service_pb2
from tritonclient.grpc.service_pb2_grpc import GRPCInferenceServiceStub


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


def deserialize_bytes_tensor(encoded_tensor: Any) -> np.ndarray:
    strs = list()
    offset = 0
    val_buf = encoded_tensor
    while offset < len(val_buf):
        length = struct.unpack_from("<I", val_buf, offset)[0]
        offset += 4
        sb = struct.unpack_from("<{}s".format(length), val_buf, offset)[0]
        offset += length
        strs.append(sb)
    return np.array(strs, dtype=np.object_)


def postprocess(response: Any) -> str:
    """Post-process response to get the text spoken"""
    if len(response.outputs) != 1:
        raise ValueError(f"Expected 1 output, got {len(response.outputs)}")

    if len(response.raw_output_contents) != 1:
        raise ValueError(
            f"Expected 1 output content, got {len(response.raw_output_contents)}"
        )

    batched_result = deserialize_bytes_tensor(response.raw_output_contents[0])
    content = np.reshape(batched_result, response.outputs[0].shape)

    output = "".join(x.decode().split(":")[-1] for x in content)
    return output
