import argparse
import numpy as np
import timm
import torch
from reprod_log import ReprodLogger


def load_npz_state(model, npz_path):
    data = np.load(npz_path, allow_pickle=False)
    sd = model.state_dict()
    for k in sd.keys():
        if k in data.files:
            sd[k] = torch.from_numpy(data[k])
    model.load_state_dict(sd, strict=False)


def main(variant, state_npz, fake_data, output):
    model = timm.create_model(variant, pretrained=False)
    if state_npz:
        load_npz_state(model, state_npz)
    model.eval()

    data = np.load(fake_data)
    x_np = data['x'].astype('float32')
    x = torch.from_numpy(x_np)

    with torch.no_grad():
        log = ReprodLogger()
        log.add('input', x_np)

        if variant.endswith('_enc'):
            enc_out = model(x)
            log.add('enc_out', enc_out.cpu().numpy())
            log.save(output)
            print(f'saved: {output}')
            return

        stem = model.conv_stem(x)
        log.add('stem', stem.cpu().numpy())

        feat = stem
        for i, stage in enumerate(model.blocks):
            feat = stage(feat)
            log.add(f'stage{i}', feat.cpu().numpy())

        msfa = model.forward_features(x)
        log.add('msfa', msfa.cpu().numpy())

        out = model.forward_head(msfa)
        log.add('out', out.cpu().numpy())

        log.save(output)
    print(f'saved: {output}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--variant', type=str, default='mobilenetv5_300m')
    parser.add_argument('--state_npz', type=str, default=None)
    parser.add_argument('--fake_data', type=str, required=True)
    parser.add_argument('--output', type=str, required=True)
    args = parser.parse_args()
    main(args.variant, args.state_npz, args.fake_data, args.output)
