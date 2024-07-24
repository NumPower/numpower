<?php
    class NormBench
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
        *     "provideArraysL1"
        * })
        */
        public function benchNormL1($params): void
        {
            \NDArray::norm($this->testArray, 1);
        }

        /**
        * @BeforeMethods("setUp")
        * @ParamProviders({
        *     "provideArraysL2"
        * })
        */
        public function benchNormL2($params): void
        {
            \NDArray::norm($this->testArray, 2);
        }

        public function provideArraysL1() {
            yield ['testArray' => \NDArray::ones([1, 100])];
            yield ['testArray' => \NDArray::ones([1, 500])];
            yield ['testArray' => \NDArray::ones([1, 1000])];
            yield ['testArray' => \NDArray::ones([10, 100])];
            yield ['testArray' => \NDArray::ones([1000, 500])];
            yield ['testArray' => \NDArray::ones([10000, 100])];
        }

        public function provideArraysL2() {
            yield ['testArray' => \NDArray::ones([10, 100])];
            yield ['testArray' => \NDArray::ones([1000, 500])];
            yield ['testArray' => \NDArray::ones([10000, 100])];
        }
    }
?>
