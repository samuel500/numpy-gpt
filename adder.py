"""
Trains a GPT to add n-digit numbers.
"""

import os
import sys

import torch
from torch.utils.data import Dataset
from torch.utils.data.dataloader import DataLoader

from nptransformer.gpt import GPT
from nptransformer.optim import Adam, SGD

import random
import time
from collections import defaultdict
import numpy as np

import torch
from torch.utils.data.dataloader import DataLoader


def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

class Trainer:

    def __init__(self, model, optim, train_dataset):
        self.model = model
        self.optimizer = optim
        self.train_dataset = train_dataset
        self.batch_size = 64
        self.num_workers = 4
        self.max_iters = None
        self.callbacks = defaultdict(list)

        self.model = model

        # variables that will be assigned to trainer class later for logging and etc
        self.iter_num = 0
        self.iter_time = 0.0
        self.iter_dt = 0.0

    def add_callback(self, onevent: str, callback):
        self.callbacks[onevent].append(callback)

    def set_callback(self, onevent: str, callback):
        self.callbacks[onevent] = [callback]

    def trigger_callbacks(self, onevent: str):
        for callback in self.callbacks.get(onevent, []):
            callback(self)

    def run(self):
        model = self.model

        # setup the dataloader
        train_loader = DataLoader(
            self.train_dataset,
            sampler=torch.utils.data.RandomSampler(self.train_dataset, replacement=True, num_samples=int(1e10)),
            shuffle=False,
            pin_memory=True,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
        )

        model.train()
        self.iter_num = 1
        self.iter_time = time.time()
        data_iter = iter(train_loader)
        while True:

            # fetch the next batch (x, y) and re-init iterator if needed
            try:
                batch = next(data_iter)
            except StopIteration:
                data_iter = iter(train_loader)
                batch = next(data_iter)
            x, y = batch

            # forward the model
            logits, self.loss = model(x, y)

            # backprop and update the parameters
            model.zero_grad()
            self.loss.backward()
            # for tensor in model.get_trainable_tensors():
            #     tensor.grad = np.clip(tensor.grad, -1, 1)
            self.optimizer.step()


            self.trigger_callbacks('on_batch_end')
            self.iter_num += 1
            tnow = time.time()
            self.iter_dt = tnow - self.iter_time
            self.iter_time = tnow

            # termination conditions
            if self.max_iters is not None and self.iter_num >= self.max_iters:
                break

# -----------------------------------------------------------------------------

class AdditionDataset(Dataset):
    """
    Creates n-digit addition problems. For example, if n=2, then an example
    addition problem would be to add 85 + 50 = 135. This problem would be
    represented as the following string for the GPT:

    "8550531"

    This is because:
    - we are discarding the + and =, which are not necessary. We just encode the digits
      of the input numbers concatenated together.
    - the result 135 is encoded backwards to make the addition easier to learn for the
      GPT model, because of how the addition algorithm works.

    As one more example, the problem 6 + 39 = 45 would be encoded as:

    "0639054"

    where you will notice that we are padding with zeros to make sure that we always
    produce strings of the exact same size: n + n + (n + 1). When n=2, this is 7.
    At test time, we will feed in an addition problem by giving the first 2n digits,
    and hoping that the GPT model completes the sequence with the next (n+1) digits
    correctly.
    """


    def __init__(self, ndigit=2, split=''):
        self.ndigit = ndigit
        self.split = split # train/test

        # split up all addition problems into either training data or test data
        assert ndigit <= 3, "the lines below would be very memory inefficient, in future maybe refactor to support"
        num = (10**ndigit)**2 # total number of possible addition problems with ndigit numbers
        rng = torch.Generator()
        # rng.manual_seed(1337)
        perm = torch.randperm(num, generator=rng)
        num_test = min(int(num*0.2), 500) # 20% of the whole dataset, or only up to 500
        self.ixes = perm[:num_test] if split == 'test' else perm[num_test:]

    def get_vocab_size(self):
        return 10 # digits 0..9

    def get_block_size(self):
        # a,b,a+b, and +1 due to potential carry overflow,
        # but then also -1 because very last digit doesn't ever plug back
        # as there is no explicit <EOS> token to predict, it is implied
        return 3*self.ndigit + 1 - 1

    def __len__(self):
        return self.ixes.nelement()

    def __getitem__(self, idx):
        ndigit = self.ndigit
        # given a problem index idx, first recover the associated a + b
        idx = self.ixes[idx].item()
        nd = 10**ndigit
        a = idx // nd
        b = idx %  nd
        # calculate the "label" of the addition problem a + b
        c = a + b
        # encode the digits of a, b, c into strings
        astr = f'%0{ndigit}d' % a
        bstr = f'%0{ndigit}d' % b
        cstr = (f'%0{ndigit+1}d' % c)[::-1] # reverse c to make addition easier
        render = astr + bstr + cstr
        dix = [int(s) for s in render] # convert each character to its token index
        # x will be input to GPT and y will be the associated expected outputs
        x = torch.tensor(dix[:-1], dtype=torch.long)
        # x = torch.tensor(dix[:-1], dtype=torch.long)

        y = torch.tensor(dix[1:], dtype=torch.long) # predict the next token in the sequence
        y[:ndigit*2-1] = -3 # we will only train in the output locations. -1 will mask loss to zero
        return x, y


class ContinuousAdditionDataset:

    def __init__(self, ndigit=2, split=''):
        self.ndigit = ndigit
        self.split = split # train/test

        # split up all addition problems into either training data or test data
        # assert ndigit <= 3, "the lines below would be very memory inefficient, in future maybe refactor to support"
        num = (10**ndigit)**2 # total number of possible addition problems with ndigit numbers
        # rng = torch.Generator()
        # rng.manual_seed(1337)
        # perm = torch.randperm(num, generator=rng)
        # num_test = min(int(num*0.2), 500) # 20% of the whole dataset, or only up to 500


        # self.ixes = perm[:num_test] if split == 'test' else perm[num_test:]

    def get_vocab_size(self):
        return 10 # digits 0..9

    def get_block_size(self):
        # a,b,a+b, and +1 due to potential carry overflow,
        # but then also -1 because very last digit doesn't ever plug back
        # as there is no explicit <EOS> token to predict, it is implied
        return 3*self.ndigit + 1 - 1

    def __len__(self):
        return 9999999999999 # self.ixes.nelement()

    def __getitem__(self, waa):
        ndigit = self.ndigit
        # given a problem index idx, first recover the associated a + b
        # idx = self.ixes[idx].item()
        nd = 10**ndigit
        a = random.randint(0, nd-1) #idx // nd
        b = random.randint(0, nd-1) # idx %  nd
        # calculate the "label" of the addition problem a + b
        c = a + b
        # encode the digits of a, b, c into strings
        astr = f'%0{ndigit}d' % a
        bstr = f'%0{ndigit}d' % b
        cstr = (f'%0{ndigit+1}d' % c)[::-1] # reverse c to make addition easier
        render = astr + bstr + cstr
        dix = [int(s) for s in render] # convert each character to its token index
        # x will be input to GPT and y will be the associated expected outputs
        x = np.array(dix[:-1]) #, dtype=torch.long)

        y = np.array(dix[1:]) #, dtype=torch.long) # predict the next token in the sequence
        y[:ndigit*2-1] = -3 # we will only train in the output locations. -3 will mask loss to zero
        return x, y

    

# -----------------------------------------------------------------------------

if __name__ == '__main__':
    # set_seed(1337)

    # construct train and test datasets
    ndigit = 6
    # train_dataset = AdditionDataset(ndigit=ndigit, split='train')
    # test_dataset  = AdditionDataset(ndigit=ndigit, split='test')
    train_dataset = ContinuousAdditionDataset(ndigit=ndigit, split='train')
    test_dataset  = ContinuousAdditionDataset(ndigit=ndigit, split='test')
    

    # construct the model
    vocab_size = train_dataset.get_vocab_size()
    block_size = train_dataset.get_block_size()
    model = GPT(n_layers=1, n_heads=4, n_embd=64, vocab_size=vocab_size, block_size=block_size)
    optim = Adam(model.get_trainable_tensors(), learning_rate=5e-4)
    # optim = SGD(model.get_trainable_tensors())

    # construct the trainer object
    trainer = Trainer(model, optim, train_dataset)

    # helper function for the evaluation of a model
    def eval_split(trainer, split, max_batches=5):
        dataset = {'train':train_dataset, 'test':test_dataset}[split]
        results = []
        mistakes_printed_already = 0
        factors = np.array([[10**i for i in range(ndigit+1)][::-1]]) #.to(trainer.device)
        loader = DataLoader(dataset, batch_size=100, num_workers=0, drop_last=False)
        for b, (x, y) in enumerate(loader):
            # x = x #.to(trainer.device)
            # isolate the first two digits of the input sequence alone
            d1d2 = x[:, :ndigit*2]
            # let the model sample the rest of the sequence

            # d1d2 to numpy array .... 
            d1d2d3 = model.generate(d1d2, ndigit+1) #, do_sample=False) # using greedy argmax, not sampling
            
            
            # isolate the last digit of the sampled sequence
            d3 = d1d2d3[:, -(ndigit+1):]
            d3 = np.flip(d3, axis=1) # reverse the digits to their "normal" order
            # decode the integers from individual digits
            d1i = (d1d2[:,:ndigit] * factors[:,1:]).sum(1)
            d2i = (d1d2[:,ndigit:ndigit*2] * factors[:,1:]).sum(1)
            d3i_pred = np.sum(d3 * factors, axis=1) #.sum(1)
            d3i_gt = d1i + d2i # manually calculate the ground truth
            # evaluate the correctness of the results in this batch
            correct = np.equal(d3i_pred, d3i_gt) #.cpu() # Software 1.0 vs. Software 2.0 fight RIGHT on this line haha
            for i in range(x.size(0)):
                results.append(int(correct[i]))
                if not correct[i] and mistakes_printed_already < 5: # only print up to 5 mistakes to get a sense
                    mistakes_printed_already += 1
                    print("GPT claims that %d + %d = %d but gt is %d" % (d1i[i], d2i[i], d3i_pred[i], d3i_gt[i]))
            if max_batches is not None and b+1 >= max_batches:
                break
        rt = torch.tensor(results, dtype=torch.float)
        print("%s final score: %d/%d = %.2f%% correct" % (split, rt.sum(), len(results), 100*rt.mean()))
        return rt.sum()

    # iteration callback
    top_score = 0
    def batch_end_callback(trainer):
        global top_score

        if trainer.iter_num % 50 == 0:
            print(f"iter_dt {trainer.iter_dt * 1000:.2f}ms; iter {trainer.iter_num}: train loss {trainer.loss.sum():.5f}")

        if trainer.iter_num % 500 == 0:
            # evaluate both the train and test score
            # train_max_batches = {1: None, 2: None, 3: None, }[ndigit] # if ndigit=2 we can afford the whole train set, ow no
            model.eval()
            with torch.no_grad():
                # train_score = eval_split(trainer, 'train', max_batches=5)
                train_score = 0
                test_score  = eval_split(trainer, 'test',  max_batches=5)
            score = train_score + test_score
            # save the model if this is the best score we've seen so far
            if score > top_score:
                top_score = score
                print(f"not saving model with new top score of {score}")
                # ckpt_path = os.path.join(config.system.work_dir, "model.pt")
                # torch.save(model.state_dict(), ckpt_path)
            # revert model to training mode
            model.train()  # for dropout, layernorm... 
        for tensor in model.get_trainable_tensors():
            tensor.grad = np.zeros_like(tensor.grad)

    trainer.set_callback('on_batch_end', batch_end_callback)

    # run the optimization
    trainer.run()
