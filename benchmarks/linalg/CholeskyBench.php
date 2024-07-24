<?php
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
            #$matrix = \NDArray::uniform([$ndim, $ndim], 0.0, 99.0);
            #$matrix_T = \NDArray::transpose($matrix);
            #$symmetric = \NDArray::divide(\NDArray::add($matrix, $matrix_T), 2);
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
