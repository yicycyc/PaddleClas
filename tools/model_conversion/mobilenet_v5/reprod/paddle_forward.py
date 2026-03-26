import argparse
import numpy as np
import paddle
from reprod_log import ReprodLogger

from ppcls.arch.backbone import MobileNetV5_300M, MobileNetV5_300M_enc, MobileNetV5_base
from ppcls.utils import logger


def build_model(variant):
    if variant == 'mobilenetv5_300m':
        return MobileNetV5_300M(class_num=0)
    if variant == 'mobilenetv5_300m_enc':
        return MobileNetV5_300M_enc()
    if variant == 'mobilenetv5_base':
        return MobileNetV5_base(class_num=1000)
    raise ValueError(f'Unsupported variant: {variant}')


def main(variant, pdparams, fake_data, output):
    logger.init_logger()
    model = build_model(variant)
    state = paddle.load(pdparams)
    model.set_state_dict(state)
    model.eval()

    data = np.load(fake_data)
    x_np = data['x'].astype('float32')
    x = paddle.to_tensor(x_np)

    with paddle.no_grad():
        log = ReprodLogger()
        log.add('input', x_np)

        if variant.endswith('_enc'):
            enc_out = model(paddle.to_tensor(x_np)).numpy()
            log.add('enc_out', enc_out)
            log.save(output)
            print(f'saved: {output}')
            return

        stem = model.conv_stem(x)
        log.add('stem', stem.numpy())

        stage_feats = []
        feat = stem
        for idx, blk in enumerate(model.blocks):
            feat = blk(feat)
            if idx in model.stage_ends:
                stage_id = len(stage_feats)
                stage_feats.append(feat)
                log.add(f'stage{stage_id}', feat.numpy())

        msfa_inputs = [stage_feats[i] for i in model.msfa_indices]
        msfa = model.msfa(msfa_inputs)
        log.add('msfa', msfa.numpy())

        out = model.forward_head(msfa)
        log.add('out', out.numpy())

        log.save(output)
    print(f'saved: {output}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--variant', type=str, default='mobilenetv5_300m')
    parser.add_argument('--pdparams', type=str, required=True)
    parser.add_argument('--fake_data', type=str, required=True)
    parser.add_argument('--output', type=str, required=True)
    args = parser.parse_args()
    main(args.variant, args.pdparams, args.fake_data, args.output)
