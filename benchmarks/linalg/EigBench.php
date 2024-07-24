<?php
    class EigBench
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
        public function benchEig($params): void
        {
            \NDArray::eig($this->testArray);
        }

        public function provideArrays() {
            yield ['testArray' => \NDArray::ones([100, 100])];
            yield ['testArray' => \NDArray::ones([500, 500])];
            yield ['testArray' => \NDArray::ones([1000, 1000])];
        }
    }
?>
