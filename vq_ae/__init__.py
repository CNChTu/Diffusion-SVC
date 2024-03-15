def get_model(args):
    if args.vqae.type == 'transformer':
        from vq_ae.transformer.model import get_model
    elif args.vqae.type == 'cnn':
        from vq_ae.wavenet.model import get_model
    else:
        raise ValueError(' [!] Unkown model type: {}'.format(args.vqae.type))
    model = get_model(args)
    return model