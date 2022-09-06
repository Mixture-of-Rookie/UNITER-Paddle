# Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import paddle
from paddle import inference
import numpy as np
from PIL import Image


class InferenceEngine(object):
    """InferenceEngine
    
    Inference engina class which contains preprocess, run, postprocess
    """

    def __init__(self, args):
        """
        Args:
            args: Parameters generated using argparser.
        Returns: None
        """
        super().__init__()
        self.args = args

        # init inference engine
        self.predictor, self.config, self.input_names, self.input_tensor, self.output_names, self.output_tensor = self.load_predictor(
            os.path.join(args.model_dir, "inference.pdmodel"),
            os.path.join(args.model_dir, "inference.pdiparams"))

    def load_predictor(self, model_file_path, params_file_path):# {{{
        """load_predictor
        initialize the inference engine
        Args:
            model_file_path: inference model path (*.pdmodel)
            model_file_path: inference parmaeter path (*.pdiparams)
        Return:
            predictor: Predictor created using Paddle Inference.
            config: Configuration of the predictor.
            input_tensor: Input tensor of the predictor.
            output_tensor: Output tensor of the predictor.
        """
        args = self.args
        config = inference.Config(model_file_path, params_file_path)
        if args.use_gpu:
            config.enable_use_gpu(1000, 0)
        else:
            config.disable_gpu()
            # The thread num should not be greater than the number of cores in the CPU.
            config.set_cpu_math_library_num_threads(4)

        # enable memory optim
        config.enable_memory_optim()
        config.disable_glog_info()

        config.switch_use_feed_fetch_ops(False)
        config.switch_ir_optim(True)

        # create predictor
        predictor = inference.create_predictor(config)

        # get input and output tensor property
        input_names = predictor.get_input_names()
        input_tensor = []
        for i in range(len(input_names)):
            input_tensor.append(predictor.get_input_handle(input_names[i]))

        output_names = predictor.get_output_names()
        output_tensor = predictor.get_output_handle(output_names[0])

        return predictor, config, input_names, input_tensor, output_names, output_tensor# }}}

    def preprocess(self, data_path):# {{{
        """preprocess
        Preprocess to the input.
        Args:
            data_path: path to the lite dataset.
        Returns: Input data after preprocess.
        """
        data = np.load(data_path)
        data_list = []
        for name in self.input_names:
            if name == 'expand_0.tmp_0':
                name = 'input_ids'
            data_list.append(data[name])
        return data_list# }}}

    def postprocess(self, x):# {{{
        """postprocess
        Postprocess to the inference engine output.
        Args:
            x: Inference engine output.
        Returns: Output data after argmax.
        """
        x = x.flatten()
        img_id = x.argmax()
        return img_id# }}}

    def run(self, x):# {{{
        """run
        Inference process using inference engine.
        Args:
            x: Input data after preprocess.
        Returns: Inference engine output
        """
        for i in range(len(self.input_tensor)):
            self.input_tensor[i].copy_from_cpu(x[i])
        self.predictor.run()
        output = self.output_tensor.copy_to_cpu()
        return output# }}}


def get_args(add_help=True):# {{{
    """
    parse args
    """
    import argparse

    def str2bool(v):
        return v.lower() in ("true", "t", "1")

    parser = argparse.ArgumentParser(
        description="PaddlePaddle Classification Training", add_help=add_help)

    parser.add_argument(
        "--model-dir", default='./static/', help="inference model dir")
    parser.add_argument(
        "--use-gpu", default=False, type=str2bool, help="use_gpu")
    parser.add_argument(
        "--batch-size", default=1, type=int, help="batch size")
    parser.add_argument(
        "--data-dir", default="./lite_data/infer")
    parser.add_argument(
        "--data-path", default="./lite_data/infer/lite_dataset.npz")
    parser.add_argument(
        "--benchmark", default=False, type=str2bool, help="benchmark")

    args = parser.parse_args()
    return args# }}}


def infer_main(args):
    """infer_main
    Main inference function.
    Args:
        args: Parameters generated using argparser.
    Returns:
        class_id: Class index of the input.
        prob: : Probability of the input.
    """
    inference_engine = InferenceEngine(args)

    # init benchmark
    if args.benchmark:
        import auto_log
        autolog = auto_log.AutoLogger(
            model_name="image-text retrieval",
            batch_size=args.batch_size,
            inference_config=inference_engine.config,
            gpu_ids="auto" if args.use_gpu else None)

    assert args.batch_size == 1, "batch size just supports 1 now."

    # enable benchmark
    if args.benchmark:
        autolog.times.start()

    # preprocess
    data = inference_engine.preprocess(args.data_path)

    if args.benchmark:
        autolog.times.stamp()

    output = inference_engine.run(data)

    if args.benchmark:
        autolog.times.stamp()

    # postprocess
    image_idx = inference_engine.postprocess(output)
    ann_file = os.path.join(args.data_dir, 'lite_dataset_images.txt')
    img_db = {}
    with open(ann_file, 'r') as f:
        txt_query = f.readline()
        for img_info in f.readlines():
            img_idx, img_name = img_info.strip().split(' ')
            img_db[int(img_idx)] = img_name

    if args.benchmark:
        autolog.times.stamp()
        autolog.times.end(stamp=True)
        autolog.report()

    print('TxT Query:')
    print(txt_query)
    print('Retreival Result:')
    print(img_db[image_idx])


if __name__ == "__main__":
    args = get_args()
    infer_main(args)
