#include <cassert>
#include <cstdio>
#include <memory>

template<size_t N>
  class shared_array
{
  public:
    /*! \pre This thread block shall be converged.
     */
    __device__
    shared_array()
    {
      construct_tags();

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

      barrier();

      destroy_tags();
    }

    __device__
    int &operator[](int i)
    {
      // gain exclusive access to tag i
      assert_atomically_obtain_tag(i, threadIdx.x);

      return m_array[i];
    }

    // XXX we probably want to return a value instead of a reference here
    //     otherwise a writer could bash the referent
    //     we should gain exclusive access to the element, make a copy, and then relinquish the lock
    __device__
    const int &operator[](int i) const
    {
      assert_ownership_or_no_owner(i, threadIdx.x);

      return m_array[i];
    }

    /*! \pre This thread block shall be converged.
     */
    __device__
    void barrier()
    {
      clear_tags_and_synchronize();
    }

    __device__
    size_t size() const
    {
      return N;
    }

  private:
    __device__
    void construct_tags()
    {
#if CUDA_ARCH < 200
      __shared__ int tags[N];
#else
      __shared__ int *tags;
      if(threadIdx.x == 0)
      {
        tags = static_cast<int*>(malloc(sizeof(int *)));
      }
#endif

      __syncthreads();
      if(threadIdx.x == 0)
      {
        m_tags = tags;
      }
      __syncthreads();

      clear_tags_and_synchronize();
    }

    __device__
    void destroy_tags()
    {
#if CUDA_ARCH < 200
      // nothing to do
#else
      if(threadIdx.x == 0)
      {
        free(tags);
      }
#endif
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
    void clear_tags_and_synchronize()
    {
      const int tag_is_clear = blockDim.x + 1;

      if(threadIdx.x == 0)
      {
        for(int i = 0; i < size(); ++i)
        {
          m_tags[i] = tag_is_clear;
        }
      }

      __syncthreads();
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
    void assert_ownership_or_no_owner(int i, int thread_idx) const
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

