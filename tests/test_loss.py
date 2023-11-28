import torch

import sincere.losses as losses

import spoof
import simgcd_losses


def test_self_sup():
    embeds = spoof.spoof_self_sup_embeds()
    old_val = simgcd_losses.old_info_nce(torch.vstack((embeds[:, 0], embeds[:, 1])))
    new_val = losses.InfoNCELoss(temperature=1)(embeds)
    assert torch.isclose(old_val, new_val)


def test_sup():
    embeds, labels = spoof.spoof_sup_embeds()
    # use default "all" contrast mode, which computes loss for all views instead of single view
    old_loss = simgcd_losses.SupConLoss()
    new_loss = losses.MultiviewSINCERELoss()
    old_val = old_loss(embeds, labels)
    new_val = new_loss(embeds, labels)
    # new loss always strictly less than old loss due to the correction of the softmax denominator
    # old_loss usually greater than 1.5
    # new loss usually much less than 0.1, but varies more from random samples
    assert old_val > new_val


def test_sup_ce():
    preds, labels = spoof.spoof_logits()
    old_val = simgcd_losses.old_sup_ce(preds, labels)
    new_val = losses.TempCELoss(0.1)(preds, labels)
    assert torch.allclose(old_val, new_val)


def test_mean_ent():
    n_labels = 4
    preds, _ = spoof.spoof_logits(n_labels)
    old_val = simgcd_losses.old_mean_ent(preds)
    new_val = losses.TempMeanEntropyLoss()(preds)
    # new loss doesn't add term to make loss positive
    assert torch.allclose(old_val, new_val + torch.log(torch.Tensor([n_labels])))
