<?php
    /**
     * @Groups({"linearAlgebra"})
     */
    class CholeskyBench
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
        public function benchCholesky($params): void
        {
            \NDArray::cholesky($this->testArray);
        }

        private function createArray($ndim) {
            $symmetric = \NDArray::ones([$ndim, $ndim]);
            for ($i=0; $i<$ndim; $i++) {
                $symmetric[$i][$i] += $ndim * 10;
            }
            $L = \NDArray::cholesky($symmetric);
            $L_T = \NDArray::transpose($L);
            $positive_definite_matrix = \NDArray::matmul($L, $L_T);
            return $positive_definite_matrix;
        }

        public function provideArrays() {
            $testSizes = array(10, 100, 1000, 10000);
            foreach ($testSizes as &$value) {
                $currTestMatrix = $this->createArray($value);
                yield ['testArray' => $currTestMatrix];
            }
        }
    }
?>
