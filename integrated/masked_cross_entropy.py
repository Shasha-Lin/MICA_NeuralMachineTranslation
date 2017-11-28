import torch
from torch.nn import functional
from torch.autograd import Variable


### Softmax on predicted words in the batch:
    # - The actuals (A) have the shape (batch_size, true classes) (Note: batch_size = no of sentences, true_class = vocabulary )
    # - The predicted (P) words (classes) have the shape (batch_size, max_lenght_in_batch, vocab size). [By max_length_in_batch I mean the length corresponding to the largest sentence in the bacth]
    # - P is now unrolled to have the shape (batch_size*max_length_in_batch, vocab_size)
    # - The indicator function is now applied. (It extracts the loss corresponding to the true label and rejects all the other loss.) Hence, P is now reduced to the size, (batch_size*max_length_in_batch, 1)
    # - P is now reshaped and converted back to the generic shape. (batch_size, max_length_in_batch)
    # - We know that every sentence in the batch doesn't have a length equal to max_length. Hence, the losses for words > (max_length of that particular sentence) are replaced with zeros.
    # - Finally, loss = (P[i,:].sum()/length).sum() for i in batch_size

def sequence_mask(sequence_length, max_len=None):
    if max_len is None:
        max_len = sequence_length.data.max()
    batch_size = sequence_length.size(0)
    seq_range = torch.range(0, max_len - 1).long()
    seq_range_expand = seq_range.unsqueeze(0).expand(batch_size, max_len)
    seq_range_expand = Variable(seq_range_expand)
    if sequence_length.is_cuda:
        seq_range_expand = seq_range_expand.cuda()
    seq_length_expand = (sequence_length.unsqueeze(1)
                         .expand_as(seq_range_expand))
    return seq_range_expand < seq_length_expand


def masked_cross_entropy(logits, target, length, use_cuda=False):
    if use_cuda:
        length = Variable(torch.LongTensor(length)).cuda()
    else:
        length = Variable(torch.LongTensor(length))
    """
    Args:
        logits: A Variable containing a FloatTensor of size
            (batch, max_len, num_classes) which contains the
            unnormalized probability for each class.
        target: A Variable containing a LongTensor of size
            (batch, max_len) which contains the index of the true
            class for each corresponding step.
        length: A Variable containing a LongTensor of size (batch,)
            which contains the length of each data in a batch.
    Returns:
        loss: An average loss value masked by the length.
    """

    # logits_flat: (batch * max_len, num_classes)
    logits_flat = logits.view(-1, logits.size(-1))
    # log_probs_flat: (batch * max_len, num_classes)
    log_probs_flat = functional.log_softmax(logits_flat)
    # target_flat: (batch * max_len, 1)
    target_flat = target.view(-1, 1)
    # losses_flat: (batch * max_len, 1)
    losses_flat = -torch.gather(log_probs_flat, dim=1, index=target_flat)
    # losses: (batch, max_len)
    losses = losses_flat.view(*target.size())
    # mask: (batch, max_len)
    mask = sequence_mask(sequence_length=length, max_len=target.size(1))
    losses = losses * mask.float()
    loss = losses.sum() / length.float().sum()
    return loss