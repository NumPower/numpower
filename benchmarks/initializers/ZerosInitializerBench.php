<?php
    class ZerosInitializerBench
    {
        /**
        * @Revs(1000)
        * @Iterations(5)
        */
        public function benchMatrix_10x100()
        {
            $ndarray = \NDArray::zeros([10, 100, 1]);
        }
        /**
        * @Revs(1000)
        * @Iterations(5)
        */
        public function benchMatrix_1000x500()
        {
            $ndarray = \NDArray::zeros([500, 1000, 1]);
        }
        /**
        * @Revs(1000)
        * @Iterations(5)
        */
        public function benchMatrix_10000x1000()
        {
            $ndarray = \NDArray::zeros([1000, 10000, 1]);
        }
        /**
        * @Revs(1000)
        * @Iterations(5)
        */
        public function benchVector_100()
        {
            $ndarray = \NDArray::zeros([100, 1, 1]);
        }
        /**
        * @Revs(1000)
        * @Iterations(5)
        */
        public function benchVector_500()
        {
            $ndarray = \NDArray::zeros([500, 1, 1]);
        }
        /**
        * @Revs(1000)
        * @Iterations(5)
        */
        public function benchVector_1000()
        {
            $ndarray = \NDArray::zeros([1000, 1, 1]);
        }
    }
?>
