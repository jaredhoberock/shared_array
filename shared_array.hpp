#include <cassert>
#include <cstdio>

template<size_t N>
  class shared_array
{
  public:
    /*! \pre This thread block shall be converged.
     */
    __device__
    shared_array()
    {
      initialize_tags();

      __shared__ int array[N];

      barrier();
      if(threadIdx.x == 0)
      {
        m_array = array;
      }
      barrier();

      construct_elements();
    }

    /*! \pre This thread block shall be converged.
     */
    __device__
    ~shared_array()
    {
      destroy_elements();
    }

    __device__
    int &operator[](int i)
    {
      // gain exclusive access to tag i
      assert_atomically_obtain_tag(i, threadIdx.x);

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
    void barrier()
    {
      clear_tags();
      __syncthreads();
    }

    __device__
    size_t size() const
    {
      return N;
    }

  private:
    __device__
    void initialize_tags()
    {
      __shared__ int tags[N];

      __syncthreads();
      if(threadIdx.x == 0)
      {
        m_tags = tags;
      }
      __syncthreads();

      clear_tags();
    }

    __device__
    void construct_elements()
    {
    }

    __device__
    void destroy_elements()
    {
    }

    __device__
    void clear_tags()
    {
      const int tag_is_clear = blockDim.x + 1;

      if(threadIdx.x == 0)
      {
        for(int i = 0; i < size(); ++i)
        {
          m_tags[i] = tag_is_clear;
        }
      }
    }

    __device__
    bool atomically_obtain_tag(int i, int thread_idx)
    {
      const int tag_is_clear = blockDim.x + 1;

      // only grab the tag if it has no other owner
      int old_tag = atomicCAS(m_tags + i, tag_is_clear, thread_idx);

      // we own the tag if we were able to obtain it or it we already owned it
      return (old_tag == tag_is_clear) || (old_tag == thread_idx);
    }

    __device__
    void assert_atomically_obtain_tag(int i, int thread_idx)
    {
      if(!atomically_obtain_tag(i, threadIdx.x))
      {
        // multiple writers
        printf("Write after write hazard detected in thread %d of block %d\n", threadIdx.x, blockIdx.x);
        assert(false);
      }
    }

    __device__
    void assert_clean(int i, int thread_idx) const
    {
      if(m_tags[i] != (blockDim.x + 1) && m_tags[i] != thread_idx)
      {
        printf("Read after write hazard detected in thread %d of block %d\n", threadIdx.x, blockIdx.x);
        assert(false);
      }
    }

    int *m_array;
    int *m_tags;
};

