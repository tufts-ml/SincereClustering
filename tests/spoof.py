import torch


def spoof_embeds(batch_size=10, embed_dim=10, n_views=2):
    # construct random normalized vectors
    embeds = torch.nn.functional.normalize(torch.rand((batch_size, embed_dim)) - .5)
    # return duplicates for each view
    return torch.stack((embeds,) * n_views, dim=1)


def spoof_self_sup_embeds(batch_size=10, embed_dim=10):
    return spoof_embeds(batch_size, embed_dim, 2)


def spoof_sup_embeds(n_labels=4, n_per_label=3, embed_dim=10):
    # use self-supervised embeds as base
    embeds = spoof_self_sup_embeds(n_labels, embed_dim)
    # construct labels with repeats next to each other
    labels = torch.arange(n_labels)
    labels = torch.repeat_interleave(labels, n_per_label, dim=0)
    # duplicate random vectors with the same label
    embeds = torch.repeat_interleave(embeds, n_per_label, dim=0)
    return embeds, labels


def spoof_logits(n_labels=4, n_per_label=3):
    # constuct logits
    preds = torch.rand((n_labels * n_per_label, n_labels)).softmax(dim=1)
    # construct labels with repeats next to each other
    labels = torch.arange(n_labels)
    labels = torch.repeat_interleave(labels, n_per_label, dim=0)
    return preds, labels
