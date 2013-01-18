struct uninitialized_t {};

struct thread_t {};

struct block_t {};

extern single_thread_t single_thread;
extern thread_block_t  thread_block;

template<size_t N>
  class shared_array
{
  public:
    /*! \pre This thread block shall be converged.
     */
    __device__
    shared_array(thread_block_t)
    {
      __shared__ int last_touched[N];
      __shared__ int array[N];

      __syncthreads();
      if(threadIdx.x == 0)
      {
        m_last_touched = last_touched;
        m_array        = impl;
      }
      __syncthreads();

      construct(block);
    }

    __device__
    int &operator[](int i)
    {
      assert_clean(i, threadIdx.x);

      // dirty location i
      m_last_touched[i] = threadIdx.x;

      return m_array[i];
    }

    __device__
    const int &operator[](int i) const
    {
      assert_clean(i, threadIdx.x);

      return m_array[i];
    }

    /*! \pre This thread block shall be converged.
     */
    __device__
    void barrier(thread_block_t)
    {
      clear_last_touched(thread_block_t);
      __syncthreads();
    }

  private:
    __device__
    void construct(thread_block_t)
    {
    }

    __device__
    void clear_last_touched(thread_block_t)
    {
      if(threadIdx.x == 0)
      {
        for(int i = 0; i < size(); ++i)
        {
          m_last_touched[i] = blockDim.x + 1;
        }
      }
    }

    __device__
    void assert_clean(int i, int thread_idx) const
    {
      if(m_last_touched[i] != (blockDim.x + 1) && m_last_touched[i] != thread_idx)
      {
        assert();
      }
    }

    int *m_array;
    int *m_last_touched;
};

