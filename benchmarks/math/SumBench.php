<?php
    class SumBench
    {
        /**
        * @var testArray
        */
        private $testArray = [1, 2, 3];

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
        public function benchSum($params): void
        {
            \NDArray::sum($this->testArray);
        }

        public function provideArrays() {
            yield ['testArray' => \NDArray::ones([1, 100])];
            yield ['testArray' => \NDArray::ones([1, 500])];
            yield ['testArray' => \NDArray::ones([1, 1000])];
            yield ['testArray' => \NDArray::ones([10, 100])];
            yield ['testArray' => \NDArray::ones([1000, 500])];
            yield ['testArray' => \NDArray::ones([10000, 100])];
        }
    }
?>
