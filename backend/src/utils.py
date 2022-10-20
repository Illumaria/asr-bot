from typing import Any, Tuple

import grpc
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
