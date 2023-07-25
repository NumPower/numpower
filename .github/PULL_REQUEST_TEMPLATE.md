#### What kind of change does this PR introduce? (Bug fix, feature, docs update, ...)

Lorem ipsum dolor sit amet, consectetur adipiscing elit. Duis nec lectus a mi gravida posuere id eget nibh. Nulla ipsum urna, efficitur vel ornare ut, fermentum nec sem. 

#### What is the current behavior? (You can also link to an open issue here)

Lorem ipsum dolor sit amet, consectetur adipiscing elit. Duis nec lectus a mi gravida posuere id eget nibh. Nulla ipsum urna, efficitur vel ornare ut, fermentum nec sem. 

#### What is the new behavior (if this is a feature change)?

Lorem ipsum dolor sit amet, consectetur adipiscing elit. Duis nec lectus a mi gravida posuere id eget nibh. Nulla ipsum urna, efficitur vel ornare ut, fermentum nec sem. 

#### Does this PR introduce a breaking change? (What changes might users need to make in their application due to this PR?)

Lorem ipsum dolor sit amet, consectetur adipiscing elit. Duis nec lectus a mi gravida posuere id eget nibh. Nulla ipsum urna, efficitur vel ornare ut, fermentum nec sem. 

#### Other information:

Lorem ipsum dolor sit amet, consectetur adipiscing elit. Duis nec lectus a mi gravida posuere id eget nibh. Nulla ipsum urna, efficitur vel ornare ut, fermentum nec sem. 

## Submission Checklist:
Due to the inherent complexity of this library, we created this checklist to remind everyone of the essential steps to have an MR approved depending on the type of change that is made.


* [ ] Have you followed the guidelines in our Contributing document?
* [ ] Have you checked to ensure there aren't other open [Pull Requests](../../../pulls) for the same update/change?
* [ ] Does your submission pass tests with ZEND_ALLOC enabled? `export USE_ZEND_ALLOC=1 && make test`
* [ ] Does your submission pass tests with ZEND_ALLOC disabled? `export NDARRAY_FREEBUFFER=1 && export USE_ZEND_ALLOC=0 && make test`

### Change to methods and operations
* [ ] Have you verified that your change does not break backwards compatibility?
* [ ] Have you updated the operation(s) tests for your changes, as applicable?
* [ ] **Optional:** Have your changes also been tested on the GPU? If you don't have a GPU available, you'll need to wait for a community member to perform the approval with a GPU
* [ ] **Optional:** Have your changes also been tested on the GPU with the `NDARRAY_VCHECK` option enabled and no VRAM memory leaks were displayed? `export NDARRAY_VCHECK=1 && make test`

### Changes to Core Features:
This include changes to: `buffer.c`, `gpu_alloc.c`, `ndarray.c`, `iterators.c` and their associated header files.
* [ ] Have you added an explanation of what your changes do and why you'd like us to include them?
* [ ] Have you written new tests for your core changes, as applicable?
* [ ] Your change does not affect the GPU, or if it does, it was tested with the GPU and the NDARRAY_VCHECK environment variable set with no VRAM memory leaks warnings
