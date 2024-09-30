<?php
    /**
     * @Groups({"linearAlgebra"})
     */
    class InvBench
    {
        /**
        * @var testArray
        */
        private $testArray = [];

        

        public function setUp(array $params): void
        {
            $this->testArray = $params['testArray'];
        }

        /**
        * @BeforeMethods("setUp")
        * @Revs(1000)
        * @Iterations(5)
        * @ParamProviders({
        *     "provideArrays"
        * })
        */
        public function benchInv($params): void
        {
            \NDArray::inv($this->testArray);
        }

        private function createArray($ndim) {
            $symmetric = \NDArray::ones([$ndim, $ndim]);
            for ($i=0; $i<$ndim; $i++) {
                $symmetric[$i][$i] += $ndim * 10;
            }
            return $symmetric;
        }


        public function provideArrays() {
            $testSizes = array(10, 100, 1000);
            foreach ($testSizes as &$value) {
                $currTestMatrix = $this->createArray($value);
                yield ['testArray' => $currTestMatrix];
            }
        }
    }
?>
