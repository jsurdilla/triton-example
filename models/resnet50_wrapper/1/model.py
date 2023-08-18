import json

# triton_python_backend_utils is available in every Triton Python model. You
# need to use this module to create inference requests and responses. It also
# contains some utility functions for extracting information from model_config
# and converting Triton input/output types to numpy types.
import triton_python_backend_utils as pb_utils

from torch.utils.dlpack import from_dlpack

import numpy as np
import requests as reqs

class TritonPythonModel:
    def initialize(self, args):
        # You must parse model_config. JSON string is not parsed here
        self.model_config = model_config = json.loads(args["model_config"])

        output0_config = pb_utils.get_output_config_by_name(model_config, "output__0")

        self.output_dtype = pb_utils.triton_string_to_numpy(
            pb_utils.get_output_config_by_name(json.loads(args["model_config"]), "output__0")["data_type"])

        # Convert Triton types to numpy types
        self.output0_dtype = pb_utils.triton_string_to_numpy(
            output0_config["data_type"]
        )

    def execute(self, requests):
        # For no reason at all except to demonstrate that you can do anything here.
        response = reqs.get("https://jsonplaceholder.typicode.com/todos/1")
        print(json.dumps(response.json(), indent=4))

        # TODO: What inputs and computations are we likely to use that we should try out?
        # - JSON string/byte array as input?
        # - JSON string/byte array as output?
        #
        # Or between TYPE_STRING and binary input, are we pretty confident any input would work?

        responses = []

        # Every Python backend must iterate over everyone of the requests
        # and create a pb_utils.InferenceResponse for each of them.
        for request in requests:
            input = pb_utils.get_input_tensor_by_name(request, "input__0")

            # Call the resnet50 model Triton is also serving.
            decoding_request = pb_utils.InferenceRequest(
                model_name="resnet50",
                requested_output_names=["output__0"],
                inputs=[input],
            )
            decoding_response = decoding_request.exec()
            if decoding_response.has_error():
                print(f"Error: {decoding_response.error().message()}")
                raise pb_utils.TritonModelException(
                    decoding_response.error().message())
            else:
                decoded_image = pb_utils.get_output_tensor_by_name(
                    decoding_response, "output__0")

            decoded_image = from_dlpack(decoded_image.to_dlpack()).clone()
            decoded_image = (decoded_image / 2 + 0.5).clamp(0, 1)
            decoded_image = decoded_image.detach().cpu().permute(0, 2, 3, 1).numpy()
            decoded_image = (decoded_image * 255).round().astype("uint8")

            inference_response = pb_utils.InferenceResponse(output_tensors=[
                pb_utils.Tensor("output__0", np.array(decoded_image, dtype=self.output_dtype))
            ])
            responses.append(inference_response)

        return responses

    def finalize(self):
        print("Cleaning up...")