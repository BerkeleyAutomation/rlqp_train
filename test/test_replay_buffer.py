import unittest
from rlqp_train import ReplayBuffer

import os
import mmap
import numpy as np
import numpy.testing as npt
from multiprocessing import Process

def f(arr):
    '''Helper for test_mmap_share'''
    for i in range(arr.shape[0]):
        arr[i] = i

class TestReplayBuffer(unittest.TestCase):
    def setUp(self):
        self.file_name = "replay_buffer_test"
        if os.path.exists(self.file_name):
            os.unlink(self.file_name)
            
    def tearDown(self):
        if os.path.exists(self.file_name):
            os.unlink(self.file_name)        
    
    def test_mmap_share(self):
        """Test that we can use mmap to create a memory buffer that we
        can share between parent and child processes.  This behavior
        should be standard, but since its a prerequisite for the replay
        buffer, we test it here.
        """
        rows = 100
        size = rows*4

        # open_flags = os.O_RDWR | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW
        # if hasattr(os, 'O_BINARY'):
        #     open_flags |= os.O_BINARY
        # fd = os.open(name, open_flags, 0o600)
        # os.ftruncate(fd, size)
        # buf = mmap.mmap(fd, 0)
        # os.close(fd)
        with open(self.file_name, "w+b") as fp:
            fp.truncate(size)
            buf = mmap.mmap(fp.fileno(), 0)

        arr = np.ndarray(shape=(rows,), dtype=np.int32, buffer=buf)
            
        p = Process(target=f, args=(arr,))
        p.start()
        p.join()

        # Make sure changes from child process are visible in
        # parent process
        for i in range(rows):
            self.assertEqual(i, arr[i])

    def test_shared_replay(self):
        '''Test that we can write observations, actions, etc. in a child
        process and see the results in the parent.
        '''
        buf = ReplayBuffer(self.file_name, obs_dim=(4,), act_dim=(1,), capacity=10)

        self.assertEqual(buf.index(), 0)
        
        obs = np.array([1,2,3,4], dtype=np.float32)
        ob2 = np.array([5,6,7,8], dtype=np.float32)
        act = np.array([9], dtype=np.float32)
        rew = 10.0
        don = 11.0

        def store():
            buf.store(obs, act, rew, ob2, don)

        # Store data using a child process
        p = Process(target=store)
        p.start()
        p.join()

        # In the parent process, we should see the update from the
        # child process.
        self.assertEqual(buf.index(), 1)

        # Since there is only 1 element in the buffer, if we request a
        # batch size of 2, we will get the same record twice.
        rng = np.random.default_rng(1)
        batch = buf.sample_batch(rng, 2)
        npt.assert_equal(batch['obs'].numpy(), np.vstack([obs, obs]))
        npt.assert_equal(batch['ob2'].numpy(), np.vstack([ob2, ob2]))
        npt.assert_equal(batch['act'].numpy(), np.vstack([act, act]))
        npt.assert_equal(batch['rew'].numpy(), np.array([rew, rew]))
        npt.assert_equal(batch['don'].numpy(), np.array([don, don]))

        # Make sure that tuples in the reply buffer are stored next to
        # each other.
        raw_buf = np.ndarray((11,), dtype=np.float32, buffer=buf._buf, offset=8*2)
        npt.assert_equal(raw_buf, np.array([1,2,3,4,5,6,7,8,9,10,11], dtype=np.float32))

    def test_store_array(self):
        '''Test storing an array of values.
        '''
        capacity = 10
        buf = ReplayBuffer(self.file_name, obs_dim=(4,), act_dim=(1,), capacity=capacity)
        
        obs = np.array([1,2,3,4], dtype=np.float32)
        ob2 = np.array([5,6,7,8], dtype=np.float32)
        act = np.array([9], dtype=np.float32)
        rew = 10.0
        don = 11.0

        m = 7
        add = np.linspace(1, m, num=m).reshape((m,1))*100
        obs = np.tile(obs, (m,1)) + add
        ob2 = np.tile(ob2, (m,1)) + add
        act = np.tile(act, (m,1)) + add

        buf.store_array(obs, act, rew, ob2, don)

        self.assertEqual(buf.index(), m)

        raw_buf = np.ndarray((capacity, 11), dtype=np.float32, buffer=buf._buf, offset=8*2)
        npt.assert_equal(raw_buf[:,0], np.array([101, 201, 301, 401, 501, 601, 701, 0, 0, 0]))

        # Check wrap-around in storage.  After writing reaching the
        # end of the array, new values should be stored continuing
        # from the front.
        obs += m*100
        ob2 += m*100
        act += m*100
        buf.store_array(obs, act, rew, ob2, don)
        self.assertEqual(buf.index(), m*2)
        npt.assert_equal(raw_buf[:,0], np.array([1101, 1201, 1301, 1401, 501, 601, 701, 801, 901, 1001]))


if "__main__" == __name__:
    unittest.main()
    

