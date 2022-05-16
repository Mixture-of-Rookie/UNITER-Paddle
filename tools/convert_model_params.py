import torch
import paddle

pretrained_torch_model = '/mnt/disk2T/Data/Research/Multi-Modal-Pretraining/2020-UNITER-ECCV/pretrained/uniter-base.pt'
converted_model_path = '/mnt/disk2T/Data/Research/Multi-Modal-Pretraining/2020-UNITER-ECCV/pretrained/uniter-base.bin'
pretrained_torch_model = torch.load(pretrained_torch_model)

pretrained_paddle_model = {}

# List of parameters to be transposed
transpose_list = ['uniter.img_embeddings.img_linear.weight',
                  'uniter.img_embeddings.pos_linear.weight',
                  'uniter.embeddings.token_type_embeddings.weight',
                  'uniter.pooler.dense.weight']

for i in range(24):
    transpose_list.append(
        'uniter.encoder.layer.{}.output.dense.weight'.format(str(i))
    )
    transpose_list.append(
        'uniter.encoder.layer.{}.attention.self.query.weight'.format(str(i))
    )
    transpose_list.append(
        'uniter.encoder.layer.{}.attention.self.key.weight'.format(str(i))
    )
    transpose_list.append(
        'uniter.encoder.layer.{}.attention.self.value.weight'.format(str(i))
    )
    transpose_list.append(
        'uniter.encoder.layer.{}.attention.output.dense.weight'.format(str(i))
    )
    transpose_list.append(
        'uniter.encoder.layer.{}.intermediate.dense.weight'.format(str(i))
    )

for k, v in pretrained_torch_model.items():
    if k in transpose_list:
        v = v.transpose(1, 0)
    pretrained_paddle_model[k] = paddle.to_tensor(v.cpu().numpy(), dtype='float32', place='cpu')

paddle.save(pretrained_paddle_model, converted_model_path)
