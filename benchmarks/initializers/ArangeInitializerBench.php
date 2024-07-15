<?php
    class ArangeInitializerBench
    {
        /**
        * @Revs(1000)
        * @Iterations(5)
        */
        public function benchRange_100()
        {
            $ndarray = \NDArray::arange(100, 0, 1);
        }
        /**
        * @Revs(1000)
        * @Iterations(5)
        */
        public function benchRange_500()
        {
            $ndarray = \NDArray::arange(500, 0, 1);
        }
        /**
        * @Revs(1000)
        * @Iterations(5)
        */
        public function benchRange_1000()
        {
            $ndarray = \NDArray::arange(1000, 0, 1);
        }
    }
?>