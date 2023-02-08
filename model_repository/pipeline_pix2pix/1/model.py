# Copyright (c) 2022, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions
# are met:
#  * Redistributions of source code must retain the above copyright
#    notice, this list of conditions and the following disclaimer.
#  * Redistributions in binary form must reproduce the above copyright
#    notice, this list of conditions and the following disclaimer in the
#    documentation and/or other materials provided with the distribution.
#  * Neither the name of NVIDIA CORPORATION nor the names of its
#    contributors may be used to endorse or promote products derived
#    from this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY
# EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
# PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
# CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
# EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
# PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
# PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
# OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
# (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

import json
import numpy as np
import torch
import triton_python_backend_utils as pb_utils

from torch import autocast
from torch.utils.dlpack import to_dlpack, from_dlpack
from transformers import CLIPTokenizer
from diffusers import LMSDiscreteScheduler, UNet2DConditionModel
from tqdm.auto import tqdm
import requests as req

from diffusers import StableDiffusionInstructPix2PixPipeline, EulerAncestralDiscreteScheduler
import PIL

class TritonPythonModel:

    def initialize(self, args):
        model_id = "timbrooks/instruct-pix2pix"
        self.pipe = StableDiffusionInstructPix2PixPipeline.from_pretrained(model_id, torch_dtype=torch.float16,
                                                                      safety_checker=None)
        self.pipe.to("cuda:0")
        self.pipe.scheduler = EulerAncestralDiscreteScheduler.from_config(self.pipe.scheduler.config)

    def execute(self, requests):

        responses = []

        for request in requests:

            inp = pb_utils.get_input_tensor_by_name(request, "prompt")
            input_text = inp.as_numpy()[0][0].decode()
            inp_image = pb_utils.get_input_tensor_by_name(request, "image")
            image_array = inp_image.as_numpy()[0][0]

            image_array = image_array.astype(np.uint8)
            print(image_array.shape)
            image = PIL.Image.fromarray(image_array)

            images = self.pipe(input_text, image=image, num_inference_steps=10, image_guidance_scale=1).images

            images_array = np.uint8(images[0])

            # Sending results
            inference_response = pb_utils.InferenceResponse(output_tensors=[
                pb_utils.Tensor(
                    "generated_image",
                    np.array(images_array, dtype=np.float32),
                )
            ])
            responses.append(inference_response)
        return responses

    def finalize(self):
        self.pipe = None

