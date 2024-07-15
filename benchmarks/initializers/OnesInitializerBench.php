<?php
    class OnesInitializerBench
    {
        /**
        * @Revs(1000)
        * @Iterations(5)
        */
        public function benchMatrix_10x100()
        {
            $ndarray = \NDArray::ones([10, 100, 1]);
        }
        /**
        * @Revs(1000)
        * @Iterations(5)
        */
        public function benchMatrix_1000x500()
        {
            $ndarray = \NDArray::ones([500, 1000, 1]);
        }
        /**
        * @Revs(1000)
        * @Iterations(5)
        */
        public function benchMatrix_10000x1000()
        {
            $ndarray = \NDArray::ones([1000, 10000, 1]);
        }
        /**
        * @Revs(1000)
        * @Iterations(5)
        */
        public function benchVector_100()
        {
            $ndarray = \NDArray::ones([100, 1, 1]);
        }
        /**
        * @Revs(1000)
        * @Iterations(5)
        */
        public function benchVector_500()
        {
            $ndarray = \NDArray::ones([500, 1, 1]);
        }
        /**
        * @Revs(1000)
        * @Iterations(5)
        */
        public function benchVector_1000()
        {
            $ndarray = \NDArray::ones([1000, 1, 1]);
        }
    }
?>
