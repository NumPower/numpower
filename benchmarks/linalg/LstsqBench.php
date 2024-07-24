<?php
    class LstsqBench
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
        public function benchlstsq($params): void
        {
            \NDArray::lstsq($this->testArray, $this->testArray);
        }

        public function provideArrays() {
            yield ['testArray' => \NDArray::ones([1, 100])];
            yield ['testArray' => \NDArray::ones([1, 500])];
            yield ['testArray' => \NDArray::ones([1000, 1])];
            yield ['testArray' => \NDArray::ones([10, 100])];
            yield ['testArray' => \NDArray::ones([50, 100])];
            yield ['testArray' => \NDArray::ones([10, 500])];
            yield ['testArray' => \NDArray::ones([50, 500])];
        }
    }
?>
