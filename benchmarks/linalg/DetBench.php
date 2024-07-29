<?php
    /**
     * @Groups({"linearAlgebra"})
     */
    class DetBench
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
        public function benchDet($params): void
        {
            \NDArray::det($this->testArray);
        }

        public function provideArrays() {
            yield ['testArray' => \NDArray::ones([1, 100])];
            yield ['testArray' => \NDArray::ones([1, 500])];
            yield ['testArray' => \NDArray::ones([1, 1000])];
            yield ['testArray' => \NDArray::ones([10, 100])];
            yield ['testArray' => \NDArray::ones([10, 500])];
            yield ['testArray' => \NDArray::ones([1000, 500])];
        }
    }
?>
