<?php
    class QrBench
    {
        /**
        * @var array
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
        public function benchQr($params): void
        {
            \NDArray::qr($this->testArray);
        }

        public function provideArrays() {
            yield ['testArray' => \NDArray::ones([100, 100])];
            yield ['testArray' => \NDArray::ones([500, 500])];
            yield ['testArray' => \NDArray::ones([1000, 500])];
            yield ['testArray' => \NDArray::ones([1000, 1000])];
            yield ['testArray' => \NDArray::ones([10000, 1000])];
        }
    }
?>
