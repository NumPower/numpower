<?php
    class IdentityInitializerBench
    {
        /**
        * @Revs(1000)
        * @Iterations(5)
        */
        public function benchIdentity_100()
        {
            $ndarray = \NDArray::identity(100);
        }
        /**
        * @Revs(1000)
        * @Iterations(5)
        */
        public function benchIdentity_500()
        {
            $ndarray = \NDArray::identity(500);
        }
        /**
        * @Revs(1000)
        * @Iterations(5)
        */
        public function benchIdentity_1000()
        {
            $ndarray = \NDArray::identity(1000);
        }
    }
?>