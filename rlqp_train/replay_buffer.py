import ctypes
#import multiprocessing as mp
import mmap
from multiprocessing import Lock
import numpy as np
import os
import time
import torch
import sys
import logging

log = logging.getLogger("shared_replay_buffer")

def combined_shape(length, shape=None):
    if shape is None:
        return (length,)
    return (length, shape) if np.isscalar(shape) else (length, *shape)

# def shared_array(shape, ctype=ctypes.c_float, dtype=np.float32):
#     log.debug(f"allocating buffer size {shape} => {np.prod(shape)}")
#     shared_arr = mp.Array(ctype, shape if np.isscalar(shape) else int(np.prod(shape)))
#     return np.frombuffer(shared_arr.get_obj(), dtype=dtype).reshape(shape)

def flat_size(shape):
    if np.isscalar(shape):
        return shape
    else:
        return int(np.prod(shape))

class ReplayBuffer:
    def __init__(self, name, obs_dim, act_dim, capacity):
        self._lock = Lock()

        obs_size = flat_size(obs_dim)
        act_size = flat_size(act_dim)

        obs_slice = slice(0, obs_size)
        ob2_slice = slice(obs_slice.stop, obs_slice.stop + obs_size)
        act_slice = slice(ob2_slice.stop, ob2_slice.stop + act_size)
        rew_slice = slice(act_slice.stop, act_slice.stop + 1)
        don_slice = slice(rew_slice.stop, rew_slice.stop + 1)
        
        row_size = don_slice.stop

        # number of bytes = two index int64s + rows of 
        n_bytes = (8*2) + (row_size * capacity * 4)

        # check if we need to create the file
        file_exists = os.path.isfile(name)
            
        if not file_exists:
            log.info(f"Allocating shared buffer size={n_bytes}")
            mode = "w+b"
        else:
            size = os.path.getsize(name)
            mode = "r+b"
            if n_bytes != size:
                raise ValueError(f"replay buffer size mismatch, expected {n_bytes}, file is {size}")
            log.info(f"Opening existing shared buffer size={n_bytes}")

        # Allocate a shared memory buffer that is backed by a named file.
        with open(name, mode) as fp:
            if not file_exists:
                fp.truncate(n_bytes)
            self._buf = mmap.mmap(fp.fileno(), 0)

        #self._shm = SharedMemory(create=not file_exists, size=n_bytes)
        #self._buf = self._shm.buf
        #self._buf = shared_mmap(name, size, create=)

        self.indexing = np.ndarray((2,), dtype=np.int64, buffer=self._buf)
        data = np.ndarray((capacity, row_size), dtype=np.float32, buffer=self._buf, offset=8*2)
        self.obs_buf = data[:,obs_slice]
        self.ob2_buf = data[:,ob2_slice]
        self.act_buf = data[:,act_slice]
        self.rew_buf = data[:,rew_slice].reshape((capacity,))
        self.don_buf = data[:,don_slice].reshape((capacity,))
        if not file_exists:
            assert self.indexing[0] == 0 # index
            assert self.indexing[1] == 0 # count of steps
            
        self.capacity = capacity

    def fill_ratio(self):
        return self.index() / self.capacity

    def index(self):
        return self.indexing[0]

    def steps_taken(self):
        with self._lock:
            return self.indexing[1]
        
    def store(self, obs, act, rew, ob2, done):
        assert np.isfinite(np.sum(obs))
        assert np.isfinite(np.sum(ob2))
        assert np.isfinite(np.sum(act))
        assert np.isfinite(rew)
        assert np.isfinite(done)
        
        i = self.index() % self.capacity
        self.obs_buf[i,:] = obs
        self.ob2_buf[i,:] = ob2
        self.act_buf[i,:] = act
        self.rew_buf[i] = rew
        self.don_buf[i] = done
        self.indexing[0] += 1
        self.indexing[1] += 1

    def store_array(self, obs, act, rew, ob2, done):
        i = self.indexing[0] % self.capacity
        n = obs.shape[0]
        assert act.shape[0] == n, "store_array: act shape wrong"
        assert ob2.shape[0] == n, "store_array: ob2 shape wrong"
        assert np.isscalar(rew), "store_array: rew is not a scalar"
        assert np.isscalar(done), "store_array: done is not a scalar"
        assert np.isfinite(np.sum(obs)), "store_array: obs is not finite"
        assert np.isfinite(np.sum(ob2)), "store_array: ob2 is not finite"
        assert np.isfinite(np.sum(act)), "store_array: act is not finite"
        assert np.isfinite(rew), "store_array: rew is not finite"
        assert np.isfinite(done), "store_array: done is not finite"
        if i+n <= self.capacity:
            self.obs_buf[i:i+n,:] = obs
            self.ob2_buf[i:i+n,:] = ob2
            self.act_buf[i:i+n,:] = act
            self.rew_buf[i:i+n] = rew
            self.don_buf[i:i+n] = done
        else:
            j = self.capacity - i
            self.obs_buf[i:self.capacity,:] = obs[0:j,:]
            self.ob2_buf[i:self.capacity,:] = ob2[0:j,:]
            self.act_buf[i:self.capacity,:] = act[0:j,:]
            self.rew_buf[i:self.capacity] = rew
            self.don_buf[i:self.capacity] = done
            m = min(n-j, self.capacity)
            self.obs_buf[0:m,:] = obs[j:j+m,:]
            self.ob2_buf[0:m,:] = ob2[j:j+m,:]
            self.act_buf[0:m,:] = act[j:j+m,:]
            self.rew_buf[0:m] = rew
            self.don_buf[0:m] = done

        self.indexing[0] += n
        self.indexing[1] += 1
        return self.indexing[0]

    def sample_batch(self, rng, batch_size):
        size = min(self.indexing[0], self.capacity)
        indexes = rng.integers(0, size, size=batch_size)
        batch = dict(
            obs=self.obs_buf[indexes, :],
            ob2=self.ob2_buf[indexes, :],
            act=self.act_buf[indexes, :],
            rew=self.rew_buf[indexes],
            don=self.don_buf[indexes])
        for k,v in batch.items():
            assert np.isfinite(np.sum(v)), "NaN in batch for " +k
        return {k: torch.as_tensor(v, dtype=torch.float32) for k,v in batch.items()}
